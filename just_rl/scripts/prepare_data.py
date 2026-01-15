import re
import os
import datasets
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs
import argparse

# To extract the solution for each prompts in the dataset
# def extract_solution(solution_str):
# ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set default output directory to match prepare_data.sh
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent
    DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'data' / 'processed')
    parser.add_argument('--local_dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    num_few_shot = 5
    data_source = 'BytedTsinghua-SIA/DAPO-Math-17k'

    # Load DAPO dataset (it uses split="train" directly, not a config name)
    dataset = datasets.load_dataset(data_source, split="train")

    def extract_solution(solution_str):
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
        assert solution is not None
        final_solution = solution.group(0)
        final_solution = final_solution.split('#### ')[1].replace(',', '')
        return final_solution

    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            # Extract data from DAPO schema:
            # - prompt: list[dict] with "content" key -> prompt[0]["content"]
            # - reward_model: dict with "ground_truth" and "style" keys
            # - extra_info: dict with "index" key for unique ID
            
            prompt_data = example.get('prompt', [])
            if not prompt_data or not isinstance(prompt_data, list) or len(prompt_data) == 0:
                # Skip invalid examples
                return None
            
            question = prompt_data[0].get('content', '')
            if not question:
                return None
            
            # Add instruction following (keeping same format as GSM8K processing)
            question = question + ' ' + instruction_following
            
            # Extract ground_truth from reward_model dict
            reward_model = example.get('reward_model', {})
            if not isinstance(reward_model, dict):
                return None
            
            ground_truth = reward_model.get('ground_truth', '')
            if not ground_truth:
                return None
            
            # Extract ID from extra_info if available, otherwise use idx
            extra_info = example.get('extra_info', {})
            if isinstance(extra_info, dict) and 'index' in extra_info:
                example_id = extra_info['index']
            else:
                example_id = idx
            
            # Keep the same output schema as before
            data = {
                "data_source": "math_dapo",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    'split': split,
                    'index': example_id
                }
            }
            return data

        return process_fn

    # Process dataset and filter out None values (invalid examples)
    # Filter checks if data_source exists (which means the example was successfully processed)
    def filter_valid(example):
        return example is not None and example.get('data_source') is not None
    
    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)
    train_dataset = train_dataset.filter(filter_valid)
    
    # For DAPO, check if test split exists, otherwise use train for both
    try:
        # Try to load test split
        test_dataset_raw = datasets.load_dataset(data_source, split="test")
        test_dataset = test_dataset_raw.map(function=make_map_fn('test'), with_indices=True)
        test_dataset = test_dataset.filter(filter_valid)
    except:
        # If no test split exists, use train as test
        test_dataset = train_dataset

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    makedirs(hdfs_dir)

    copy(src=local_dir, dst=hdfs_dir)