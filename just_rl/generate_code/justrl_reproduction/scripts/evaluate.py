import argparse
import os
import sys
import json
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verifiers import MathRuleVerifier
from data.prompt_utils import apply_chat_template

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate JustRL model on AIME/MATH-500")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint or HF hub ID")
    parser.add_argument("--dataset", type=str, default="math500", choices=["math500", "aime2024"], help="Dataset to evaluate on")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_model_len", type=int, default=16384, help="Max context length")
    parser.add_argument("--max_tokens", type=int, default=15360, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.jsonl", help="Output file for results")
    return parser.parse_args()

def load_eval_dataset(dataset_name: str) -> List[Dict[str, str]]:
    """
    Load and normalize dataset into a list of dicts with 'question' and 'ground_truth'.
    """
    data = []
    
    if dataset_name == "math500":
        # HuggingFaceH4/MATH-500
        print("Loading MATH-500 dataset...")
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for item in ds:
            data.append({
                "question": item["problem"],
                "ground_truth": item["answer"]
            })
            
    elif dataset_name == "aime2024":
        # AI-MO/aimo-validation-aime-2024 or similar
        # Using a known source or fallback
        print("Loading AIME 2024 dataset...")
        try:
            ds = load_dataset("AI-MO/aimo-validation-aime-2024", split="train") # Usually 'train' contains the validation set for this repo
        except:
            print("Could not load AI-MO/aimo-validation-aime-2024, trying alternative...")
            # Fallback or error
            raise ValueError("Could not load AIME 2024 dataset. Please ensure access or specify correct path.")
            
        for item in ds:
            data.append({
                "question": item["problem"],
                "ground_truth": item["answer"]
            })
            
    return data

def main():
    args = parse_args()
    
    # 1. Initialize Verifier
    verifier = MathRuleVerifier()
    
    # 2. Load Dataset
    dataset = load_eval_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples from {args.dataset}")
    
    # 3. Prepare Prompts
    prompts = [apply_chat_template(item["question"]) for item in dataset]
    
    # 4. Initialize vLLM
    print(f"Initializing vLLM with model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.9
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1.0 if args.temperature == 0 else 0.95,
    )
    
    # 5. Generate
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 6. Evaluate
    results = []
    correct_count = 0
    
    print("Verifying answers...")
    for i, output in enumerate(tqdm(outputs)):
        generated_text = output.outputs[0].text
        ground_truth = dataset[i]["ground_truth"]
        question = dataset[i]["question"]
        
        # Extract answer and verify
        extracted_answer = verifier.extract_answer(generated_text)
        
        # If extraction fails, reward is 0
        if extracted_answer is None:
            is_correct = False
            reward = 0.0
        else:
            reward = verifier.verify(generated_text, ground_truth)
            is_correct = (reward == 1.0)
            
        if is_correct:
            correct_count += 1
            
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct
        })
    
    # 7. Calculate Metrics
    accuracy = correct_count / len(dataset)
    print(f"\nResults for {args.dataset}:")
    print(f"Accuracy (Pass@1): {accuracy:.2%}")
    
    # 8. Save Results
    with open(args.output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
