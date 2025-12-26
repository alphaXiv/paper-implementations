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
Agent-specific metric utilities.
Extended from verl with additional metrics for process rewards, turns, accuracy, and format.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import _compute_response_info


def dump_generations(
    inputs: List[str],
    outputs: List[str],
    scores: List[float],
    reward_extra_infos_dict: Dict[str, List[Any]],
    dump_path: str,
    global_steps: int,
):
    """Dump rollout/validation samples as JSONL.
    
    Args:
        inputs: List of input prompts
        outputs: List of generated outputs
        scores: List of scores for each generation
        reward_extra_infos_dict: Dictionary of extra reward information
        dump_path: Directory path to save the JSONL file
        global_steps: Current global training step
    """
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{global_steps}.jsonl")

    n = len(inputs)
    base_data = {
        "input": inputs,
        "output": outputs,
        "score": scores,
        "step": [global_steps] * n,
    }

    for k, v in reward_extra_infos_dict.items():
        if len(v) == n:
            base_data[k] = v

    with open(filename, "w") as f:
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Dumped generations to {filename}")


def log_val_generations(
    inputs: List[str],
    outputs: List[str],
    scores: List[float],
    logger_config: str,
    validation_generations_logger,
    global_steps: int,
    num_generations_to_log: int,
):
    """Log a table of validation samples to the configured logger (wandb or swanlab).
    
    Args:
        inputs: List of input prompts
        outputs: List of generated outputs
        scores: List of scores for each generation
        logger_config: The logger type/config ('wandb' or 'swanlab')
        validation_generations_logger: The ValidationGenerationsLogger instance
        global_steps: Current global training step
        num_generations_to_log: Number of generations to log (0 to disable)
    """
    if num_generations_to_log == 0:
        return

    # Create tuples of (input, output, score) and sort by input text
    samples = list(zip(inputs, outputs, scores))
    samples.sort(key=lambda x: x[0])  # Sort by input text

    # Use fixed random seed for deterministic shuffling
    rng = np.random.RandomState(42)
    rng.shuffle(samples)

    # Take first N samples after shuffling
    samples = samples[:num_generations_to_log]

    # Log to each configured logger
    validation_generations_logger.log(logger_config, samples, global_steps)


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """Compute data metrics with agent-specific extensions.
    
    Extended from verl to include:
    - Process rewards metrics
    - Turns metrics  
    - Accuracy metrics
    - Format score metrics
    """
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    # Calculate process rewards metrics if enabled
    process_rewards_metrics = {}
    if "process_rewards" in batch.batch:
        process_rewards = batch.batch["process_rewards"]
        valid_process_rewards = torch.masked_select(process_rewards, batch.batch["attention_mask"][:, -process_rewards.shape[1]:].bool())
        nonzero_process_rewards = valid_process_rewards[valid_process_rewards != 0]
        
        if len(nonzero_process_rewards) > 0:
            process_rewards_metrics = {
                "critic/process_rewards/mean": torch.mean(nonzero_process_rewards).detach().item(),
                "critic/process_rewards/max": torch.max(nonzero_process_rewards).detach().item(),
                "critic/process_rewards/min": torch.min(nonzero_process_rewards).detach().item(),
                "critic/process_rewards/count": len(nonzero_process_rewards),
            }
        else:
            process_rewards_metrics = {
                "critic/process_rewards/mean": 0.0,
                "critic/process_rewards/max": 0.0,
                "critic/process_rewards/min": 0.0,
                "critic/process_rewards/count": 0,
            }

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    turns = batch.batch["turns"].float()

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # process rewards metrics
        **process_rewards_metrics,
        # turns
        'turns/mean': torch.mean(turns).detach().item(),
        'turns/max': torch.max(turns).detach().item(),
        'turns/min': torch.min(turns).detach().item(),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    if "acc" in batch.non_tensor_batch:
        acc = batch.non_tensor_batch["acc"]
        metrics.update(
            {
                # answer score
                "critic/acc/mean": np.mean(acc),
                "critic/acc/max": np.max(acc),
                "critic/acc/min": np.min(acc),
            }
        )

    if "format" in batch.non_tensor_batch:
        format_score = batch.non_tensor_batch["format"]
        metrics.update(
            {
                # format score
                "critic/format/mean": np.mean(format_score),
                "critic/format/max": np.max(format_score),
                "critic/format/min": np.min(format_score),
            }
        )
    
    return metrics
