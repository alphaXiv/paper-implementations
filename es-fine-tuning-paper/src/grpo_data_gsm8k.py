# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8K dataset to parquet format for VERL GRPO training
Reserves last 200 samples from TEST dataset for final evaluation
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/gsm8k-0.1")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_split", type=float, default=0.1, 
                        help="Fraction of train dataset to use for training")
    parser.add_argument("--test_samples", type=int, default=200,
                        help="Number of samples to reserve from TEST dataset for final evaluation")

    args = parser.parse_args()

    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction_following
            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data
        return process_fn

    # Process both datasets
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Reserve last N samples from TEST dataset for final evaluation
    test_total = len(test_dataset)
    reserved_test_size = min(args.test_samples, test_total)
    
    # Split test dataset: everything except last N samples can be used for validation
    test_for_val = test_dataset.select(range(test_total - reserved_test_size))
    
    # Reserved test set (last N samples from test dataset)
    reserved_test = test_dataset.select(range(test_total - reserved_test_size, test_total))
    
    # Create training data from train dataset
    train_total = len(train_dataset)
    train_size = max(1, int(train_total * args.train_split))
    train_data = train_dataset.select(range(train_size))
    
    # Combine remaining train data with available test data for validation
    train_for_val = train_dataset.select(range(train_size, train_total))
    available_data = datasets.concatenate_datasets([train_for_val, test_for_val])
    
    # Use combined data as validation
    val_data = available_data

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_data.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_data.to_parquet(os.path.join(local_dir, "validation.parquet"))
    reserved_test.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"\n{'='*60}")
    print(f"GSM8K Data Split Summary ({args.train_split*100:.0f}% of train dataset)")
    print(f"{'='*60}")
    print(f"Original GSM8K train dataset: {train_total} samples")
    print(f"Original GSM8K test dataset: {test_total} samples")
    print(f"\nData splits:")
    print(f"  Train: {len(train_data)} samples ({args.train_split*100:.0f}% of train dataset)")
    print(f"  Validation: {len(val_data)} samples (remaining train + test except last {reserved_test_size})")
    print(f"    - From train: {len(train_for_val)} samples")
    print(f"    - From test: {len(test_for_val)} samples")
    print(f"  Test (reserved): {len(reserved_test)} samples (LAST {reserved_test_size} from test dataset)")
    print(f"\nFiles saved to: {local_dir}")
    print(f"  - train.parquet: {len(train_data)} samples")
    print(f"  - validation.parquet: {len(val_data)} samples")
    print(f"  - test.parquet: {len(reserved_test)} samples (RESERVED FOR FINAL EVAL ONLY)")
    print(f"{'='*60}\n")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
