"""
Prompt utilities for JustRL reproduction.
Handles the specific prompt suffix formatting required by the recipe.
"""

def apply_chat_template(question: str) -> str:
    """
    Applies the specific prompt template used in JustRL.
    
    Format: "{Question}\nPlease reason step by step, and put your final answer within \\boxed{}."
    
    Args:
        question (str): The input math question.
        
    Returns:
        str: The formatted prompt.
    """
    # The paper specifies a simple suffix without complex system prompts
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    return f"{question}{suffix}"

def extract_question(prompt: str) -> str:
    """
    Extracts the original question from a formatted prompt.
    
    Args:
        prompt (str): The formatted prompt.
        
    Returns:
        str: The original question.
    """
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    if prompt.endswith(suffix):
        return prompt[:-len(suffix)]
    return prompt
