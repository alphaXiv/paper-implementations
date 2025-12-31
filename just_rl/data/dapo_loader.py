import os
import argparse
import pandas as pd
from datasets import load_dataset
import sys

# Import from the same package
# When run as a script, PYTHONPATH should include the project root
try:
    from data.prompt_utils import apply_chat_template
except ImportError:
    # Fallback: try relative import (when run as module)
    from .prompt_utils import apply_chat_template

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
        # Extract data according to DAPO dataset schema:
        # - prompt: list[dict] with "content" key -> prompt[0]["content"]
        # - reward_model: dict with "ground_truth" and "style" keys
        # - extra_info: dict with "index" key for unique ID
        
        # Extract prompt content (it's a list with a dict containing "content")
        prompt_data = example.get('prompt', [])
        if not prompt_data or not isinstance(prompt_data, list) or len(prompt_data) == 0:
            print(f"Warning: Skipping example {i} - missing or invalid prompt field")
            continue
        
        prompt_content = prompt_data[0].get('content', '')
        if not prompt_content:
            print(f"Warning: Skipping example {i} - empty prompt content")
            continue
        
        # Extract ground truth from reward_model dict
        reward_model = example.get('reward_model', {})
        if not isinstance(reward_model, dict):
            print(f"Warning: Skipping example {i} - invalid reward_model field")
            continue
        
        ground_truth = reward_model.get('ground_truth', '')
        if not ground_truth:
            print(f"Warning: Skipping example {i} - missing ground_truth")
            continue
        
        # Extract ID from extra_info if available, otherwise generate one
        extra_info = example.get('extra_info', {})
        if isinstance(extra_info, dict) and 'index' in extra_info:
            example_id = extra_info['index']
        else:
            example_id = f"dapo_{i}"
        
        # The prompt_content may already contain instructions, but we'll apply our chat template
        # to ensure consistency with the JustRL recipe format
        # If the prompt already has the suffix, apply_chat_template will add it again (harmless duplication)
        formatted_prompt = apply_chat_template(prompt_content)

        # Structure for the Trainer
        # We store 'prompt' for generation and 'ground_truth' for the verifier.
        data_list.append({
            "prompt": formatted_prompt,
            "ground_truth": ground_truth,
            "id": example_id
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
