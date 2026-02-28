"""
Reward function for GSM8K Task
This file provides a reward function for evaluating model responses on GSM8K math problems.

The reward function checks:
1. Format: Whether the response contains the expected "#### <answer>" format
2. Correctness: Whether the extracted answer matches the ground truth

Usage:
    from src.gsm8k_reward import reward_function
    reward_dict = reward_function(response, ground_truth="42")
"""

import re
from typing import Any, Dict, Optional


def extract_solution(solution_str: str, method: str = "flexible") -> Optional[str]:
    """
    Extract the numerical answer from GSM8K response format.
    
    Args:
        solution_str: The model's response string
        method: "strict" requires "#### " format, "flexible" extracts any number
        
    Returns:
        The extracted answer as a string, or None if no answer found
    """
    assert method in ["strict", "flexible"]

    if method == "strict":
        # Require the "#### " format (tests formatting)
        solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
        if solution is None:
            return None
        final_answer = solution.group(1).replace(",", "").replace("$", "")
        return final_answer
    
    elif method == "flexible":
        # Find all numbers in the response and take the last one
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        if len(answer) == 0:
            return None
        
        invalid_str = ["", "."]
        final_answer = None
        # Find the last valid number
        for final_answer in reversed(answer):
            if final_answer not in invalid_str:
                final_answer = final_answer.replace(",", "").replace("$", "")
                break
        
        return final_answer


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the GSM8K format with "#### <answer>"
    
    Args:
        response: The model's response string
        end_token: Optional end token to strip from response
        
    Returns:
        1.0 if proper format is found, 0.0 otherwise
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    # Check for "#### " format
    format_regex = r"#### (\-?[0-9\.\,]+)"
    format_match = re.search(format_regex, response)

    if format_match:
        return 1.0
    
    return 0.0


def answer_reward_function(response: str, ground_truth: str) -> float:
    """
    Checks if the extracted answer matches the ground truth.
    
    Args:
        response: The model's response string
        ground_truth: The correct answer as a string
        
    Returns:
        1.0 if answer is correct, 0.0 otherwise
    """
    # Extract answer using flexible method (more forgiving)
    answer = extract_solution(response, method="flexible")
    
    if answer is None:
        return 0.0
    
    # Compare numerical values
    try:
        answer_float = float(answer)
        gt_float = float(ground_truth.replace(",", "").replace("$", ""))
        
        # Use small epsilon for floating point comparison
        if abs(answer_float - gt_float) < 1e-5:
            return 1.0
    except (ValueError, AttributeError):
        # Fallback to string comparison if conversion fails
        if answer == ground_truth.replace(",", "").replace("$", ""):
            return 1.0
    
    return 0.0


def reward_function(
    response: str,
    ground_truth: str,
    end_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Reward function for GSM8K Tasks.
    
    Total reward = 0.1 * format_reward + answer_reward
    
    Args:
        response: The model's response string
        ground_truth: The correct answer
        end_token: Optional end token to strip from response
        
    Returns:
        Dictionary containing:
            - reward: Total reward (float between 0 and 1.1)
            - reward_info: Breakdown of format_reward and answer_reward
    """
    format_reward = format_reward_function(response, end_token)
    answer_reward = answer_reward_function(response, ground_truth)
    
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
