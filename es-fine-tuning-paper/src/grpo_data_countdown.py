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
Preprocess the Countdown dataset to parquet format for VERL GRPO training
Reserves last N samples for final evaluation
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from verl.utils.hdfs_io import copy, makedirs


SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    if think_match:
        reward += 0.1

    if answer_match:
        reward += 0.5

    return reward


def answer_reward_function(
    response: str, numbers: List[int] = None, target: int = None
) -> float:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1)
    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # Check if the answer evaluates to the target
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        pass

    return 0.0


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/countdown-0.4")
    parser.add_argument("--json_file", default="../countdown/data/countdown.json")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_split", type=float, default=0.4, 
                        help="Fraction of available data to use for training")
    parser.add_argument("--test_samples", type=int, default=200,
                        help="Number of samples to reserve for final evaluation")

    args = parser.parse_args()

    data_source = "countdown_task"

    # Load the countdown JSON data
    with open(args.json_file, 'r') as f:
        countdown_data = json.load(f)

    print(f"Loaded {len(countdown_data)} countdown tasks")

    # Process each example to VERL format
    processed_data = []
    for idx, example in enumerate(countdown_data):
        numbers = example["numbers"]
        target = float(example["target"])  # Use float to handle decimal targets
        solution = example.get("solution", "")
        
        # Create the user message
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
        
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
                {
                    "role": "assistant",
                    "content": RESPONSE_PROMPT,
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "numbers": numbers,
                    "target": target,
                }
            },
            "extra_info": {
                "index": idx,
                "numbers": numbers,
                "target": target,
                "solution": solution,
            },
        }
        processed_data.append(data)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(processed_data)
    
    total_samples = len(df)
    reserved_test_size = min(args.test_samples, total_samples)
    
    # Reserve last N samples for final evaluation
    available_data = df.iloc[:total_samples - reserved_test_size]
    reserved_test = df.iloc[total_samples - reserved_test_size:]
    
    # Split available data into train and validation
    available_size = len(available_data)
    train_size = max(1, int(available_size * args.train_split))
    
    train_df = available_data.iloc[:train_size]
    val_df = available_data.iloc[train_size:]

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Save to parquet files
    os.makedirs(local_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_df.to_parquet(os.path.join(local_dir, "validation.parquet"))
    reserved_test.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"\n{'='*60}")
    print(f"Countdown Data Split Summary ({args.train_split*100:.0f}% of available data)")
    print(f"{'='*60}")
    print(f"Total countdown tasks: {total_samples} samples")
    print(f"\nData splits:")
    print(f"  Available for training/validation: {available_size} samples")
    print(f"  Train: {len(train_df)} samples ({args.train_split*100:.0f}% of available)")
    print(f"  Validation: {len(val_df)} samples ({(1-args.train_split)*100:.0f}% of available)")
    print(f"  Test (reserved): {len(reserved_test)} samples (LAST {reserved_test_size} from dataset)")
    print(f"\nFiles saved to: {local_dir}")
    print(f"  - train.parquet: {len(train_df)} samples")
    print(f"  - validation.parquet: {len(val_df)} samples")
    print(f"  - test.parquet: {len(reserved_test)} samples (RESERVED FOR FINAL EVAL ONLY)")
    print(f"{'='*60}\n")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
