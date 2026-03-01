#!/usr/bin/env python3
"""
Evaluation script for fine-tuned models on GSM8K and Countdown test sets.
This script loads a saved model checkpoint and evaluates it on the reserved test set.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def extract_solution(solution_str):
    """Extract numerical solution from GSM8K answer format."""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def extract_answer_from_response(response: str, task_type: str = "gsm8k") -> str:
    """Extract answer from model response based on task type."""
    if task_type == "gsm8k":
        # Look for #### format answer - find ALL matches and take the LAST one
        matches = list(re.finditer(r"#### (\-?[0-9\\.\\,]+)", response))
        if matches:
            # Take the last match (the final answer)
            return matches[-1].group(1).replace(",", "")
    elif task_type == "countdown":
        # Look for <answer> tags - find ALL matches and take the LAST one
        matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
    return None


def evaluate_gsm8k_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth for GSM8K."""
    if predicted is None or ground_truth is None:
        return False
    
    try:
        pred_float = float(predicted.replace(",", ""))
        gt_float = float(ground_truth.replace(",", ""))
        return abs(pred_float - gt_float) < 1e-5
    except:
        return False


def evaluate_countdown_answer(response: str, numbers: List[int], target: float) -> bool:
    """Check if countdown answer is correct."""
    answer_regex = r"<answer>(.*?)</answer>"
    all_matches = re.findall(answer_regex, response, re.DOTALL)
    if not all_matches:
        return False

    # Take the LAST answer tag (final answer)
    answer_content = all_matches[-1].strip()
    if not answer_content:
        return False

    # Check if the answer uses only allowed characters
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return False

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return False

    # Check if the answer evaluates to the target
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return True
    except:
        pass

    return False


def load_model(model_path: str, base_model: str = None, tokenizer_path: str = None, device: str = "cuda"):
    """Load the fine-tuned model (base + LoRA adapter if applicable)."""
    print(f"Loading model from {model_path}...")
    
    
    # Load full model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    # Use custom tokenizer path if provided, otherwise use model_path
    if tokenizer_path is not None:
        print(f"Loading custom tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages: List[Dict], max_new_tokens: int = 512, device: str = "cuda"):
    """Generate response from model using chat template."""
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,  # Disable temperature when not sampling
            top_p=None,  # Disable top_p when not sampling
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (exclude the input prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def evaluate_dataset(
    model,
    tokenizer,
    test_file: str,
    task_type: str = "gsm8k",
    max_samples: int = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Evaluate model on a test dataset."""
    
    # Load test data
    df = pd.read_parquet(test_file)
    
    if max_samples:
        df = df.head(max_samples)
    
    total = len(df)
    correct = 0
    results = []
    
    print(f"\nEvaluating on {total} samples from {test_file}...")
    
    for idx, row in tqdm(df.iterrows(), total=total):
        # Get messages from the data (this matches the training format)
        messages = row['prompt']
        
        # Generate response using the same format as training
        response = generate_response(model, tokenizer, messages, device=device)
        
        # Evaluate based on task type
        is_correct = False
        ground_truth_value = None
        extracted_answer = None
        
        if task_type == "gsm8k":
            ground_truth = row['reward_model']['ground_truth']
            ground_truth_value = ground_truth
            extracted_answer = extract_answer_from_response(response, "gsm8k")
            is_correct = evaluate_gsm8k_answer(extracted_answer, ground_truth)
        elif task_type == "countdown":
            extra_info = row['extra_info']
            numbers = extra_info['numbers']
            target = extra_info['target']
            ground_truth_value = extra_info["solution"]
            extracted_answer = extract_answer_from_response(response, "countdown")
            is_correct = evaluate_countdown_answer(response, numbers, target)
        
        if is_correct:
            correct += 1
        
        # Store the prompt as text for readability in results
        prompt_text = ""
        for msg in messages:
            if msg['role'] == 'system':
                prompt_text += f"System: {msg['content']}\n\n"
            elif msg['role'] == 'user':
                prompt_text += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt_text += f"Assistant: {msg['content']}"
        
        results.append({
            'index': idx,
            'prompt': prompt_text,
            'response': response,
            'extracted_answer': extracted_answer,
            'correct': is_correct,
            'ground_truth': ground_truth_value
        })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model checkpoint or LoRA adapter")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model path (required if model_path is LoRA adapter)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to custom tokenizer (optional, if different from model/base_model)")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test parquet file")
    parser.add_argument("--task_type", type=str, choices=["gsm8k", "countdown"], 
                        default="gsm8k",
                        help="Type of task to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (default: all)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save detailed results (JSON)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model, args.tokenizer_path, args.device)
    
    # Evaluate
    eval_results = evaluate_dataset(
        model,
        tokenizer,
        args.test_file,
        args.task_type,
        args.max_samples,
        args.device
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results - {args.task_type.upper()}")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Total samples: {eval_results['total']}")
    print(f"Correct: {eval_results['correct']}")
    print(f"Accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # Save detailed results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Detailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
