import re
from mathruler.grader import extract_boxed_content, grade_answer

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def compute_score_format(solution_str):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None:
        return 0.0
    
    try:
        # Perfect format match for the new structure
        # First <|im_start|>assistant should have <think> and possibly <tool_call>
        # Then <|im_start|>tool with <tool_response> (can repeat with assistant/tool pairs)
        # Final <|im_start|>assistant with the answer and <|im_end|>
        
        # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0
        
        # If no blocks found, return 0
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0
        
        # Check the last assistant block contains <answer> tags
        last_assistant_block = assistant_blocks[-1]
        think_answer_match = re.search(r'<answer>.*\\boxed\{.*\}.*</answer>$', last_assistant_block, re.DOTALL)
        if think_answer_match:
            format_reward = 1.0
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0
    
    return format_reward


def compute_score_answer(solution_str: str, ground_truth: str) -> float:
    assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
    if len(assistant_blocks) == 0:
        return 0.0
    last_assistant_block = assistant_blocks[-1]

    answer = extract_solution(last_assistant_block)
    if answer is None:
        return 0.0
    answer = extract_boxed_content(answer)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score_format_answer(solution_str, ground_truth):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        if format_reward == 1.0:
            return answer_reward
        else:
            return -1.0
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return -1.0
