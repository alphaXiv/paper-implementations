"""
Alternative: Custom reward function for Countdown Task
This file can be used with VERL's custom_reward_function config

Usage in config:
custom_reward_function:
  path: /path/to/countdown_reward.py
  name: countdown_reward_function
  reward_kwargs: {}
"""

import re
from typing import Any, Dict, List, Optional
from verl import DataProto


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """Checks if the response follows the format <think>...</think><answer>...</answer>"""
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


def answer_reward_function(response: str, numbers: List[int], target: int) -> float:
    """Checks if the answer uses all numbers exactly once and evaluates to the target"""
    answer_regex = r"<answer>(.*?)<\/answer>"
    all_matches = re.findall(answer_regex, response, re.DOTALL)
    
    if not all_matches:
        return 0.0

    answer_content = all_matches[-1].strip()

    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        pass

    return 0.0


def countdown_reward_function(data_source: str, solution_str: str, ground_truth: Dict[str, Any], 
                              extra_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Custom reward function for countdown task compatible with VERL's custom_reward_function mechanism.
    
    This function signature matches what VERL expects from compute_score functions.
    """
    numbers = ground_truth.get("numbers", [])
    target = ground_truth.get("target", 0)
    
    if isinstance(target, str):
        target = float(target)  # Handle decimal targets from division
    
    format_reward = format_reward_function("<think>" + solution_str)
    answer_reward = answer_reward_function(solution_str, numbers, target)
    
    total_reward = format_reward * 0.1 + answer_reward
    
    return {
        "score": total_reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }
