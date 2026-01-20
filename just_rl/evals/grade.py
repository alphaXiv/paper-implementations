
from eval_utils import grade_answer_verl, grade_answer_with_error_type, extract_answer
import json
import pandas as pd
from pathlib import Path
import re

NAME     = "Qwen2.5-0.5B" # "JustRL-Nemotron-1.5B"
EVAL_DIR = Path(f"justrl_eval_outputs/{NAME}")
OUTPUT_FILE = EVAL_DIR / "grade.jsonl"

length_tokenizer = None

def get_len(seq):
    return len(length_tokenizer.encode(seq))

def get_diverse_score(sequences, n=4):
    """
    calculate the Distinct-n score。

    sequences: List[str] response list
    n: int, n-gram default=4
    """
    distinct_ngrams = set()
    total_ngrams = 0

    for seq in sequences:
        # more accurate n-gram
        # tokens = nltk.word_tokenize(seq)
        tokens = seq.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            distinct_ngrams.add(ngram)
            total_ngrams += 1

    return len(distinct_ngrams) / total_ngrams if total_ngrams > 0 else 0

def process_jsonl_file(file_name):
    """
    Process a JSONL file and dynamically handle the number of problems.
    Expected format: Each line is a JSON object with:
    - example_id: int
    - question: str
    - answer: str (ground truth)
    - seed: int
    - response: str (model output)
    
    Multiple lines can have the same example_id (when n > 1 in filename).
    """
    results = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            example_id = int(data["example_id"])
            # Ensure the list is large enough
            while len(results) <= example_id:
                results.append({"gt": None, "responses": []})
            # Extract ground truth and response
            gt = data["answer"]
            response = data["response"]
            # Store ground truth (will be overwritten if multiple lines have same example_id, but should be same)
            results[example_id]["gt"] = gt
            results[example_id]["responses"].append(response)
    return results

def parse_hyperparameters_from_filename(filename):
    """
    Parse hyperparameters from the filename.
    Example filename format: {taskname}_t{temperature}_p{top_p}_n{n}-MNT{max_tokens}.jsonl
    """
    match = re.search(r"_t(?P<temperature>[\d.]+)_p(?P<top_p>[\d.]+)_n(?P<n>\d+)-MNT(?P<max_tokens>\d+)",
                      filename)
    return match.groupdict() if match else {}

def grade_file(file_path):
    """
    Grade a single file and return the results.
    """
    hyperparams = parse_hyperparameters_from_filename(file_path.name)
    if not hyperparams:
        print(f"Skipping file with unrecognized format: {file_path}")
        return None

    task_name = file_path.stem.split("_")[0]
    hyperparams["task_name"] = task_name

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
        num_pred = len(df["responses"][0])
    elif file_path.suffix == ".jsonl":
        df = process_jsonl_file(file_path)
        if len(df) == 0:
            print(f"Warning: No data found in {file_path}")
            return None
        num_pred = len(df[0]["responses"]) if len(df[0]["responses"]) > 0 else 1
    else:
        print(f"Unsupported file format: {file_path.suffix}")
        return None

    results = {
        "hyperparameters": hyperparams,
        "mean_score": 0,
        "distinct_4gram": 0,
        "best_score": 0,
        "solve_none": 0,
        "solve_all": 0,
        "avg_output_length": 0,
        "format_error_rollouts": 0,
        "soft_mean_score": 0,
        "soft_best_score": 0,
        "soft_solve_none": 0,
        "soft_solve_all": 0,
    }

    diverse = []
    avg_scores = []
    best = []
    solve_none = 0
    solve_all = 0
    without_boxed = 0
    response_lengths = []
    incorrect_data = []  # List to store incorrect responses and ground truths

    rule_based_scores = []  # Store strict rule-based scores
    soft_scores = []  # Store soft grading scores

    for i in range(len(df)):
        if file_path.suffix == ".jsonl":
            responses = df[i]["responses"]
            gt = df[i]["gt"]
        else:  # parquet format
            responses = df["responses"][i]
            gt = df["reward_model"][i]["ground_truth"]

        responses_list = [str(response) for response in responses]
        if length_tokenizer:
            response_lengths += [get_len(response) for response in responses_list]
        else:
            response_lengths += [len(response) for response in responses_list]
        
        # Count responses where strict grader couldn't parse an answer (extract_answer returns None)
        for response in responses_list:
            parsed_answer = extract_answer(response)
            if parsed_answer is None:
                without_boxed += 1

        # Use strict rule-based verifier for scoring, but also check soft grading for error categorization
        for response in responses_list:
            strict_score, soft_score, error_type = grade_answer_with_error_type(response, gt)
            rule_based_scores.append(strict_score)
            soft_scores.append(soft_score)  # Track soft scores
            
            # Add incorrect answers to incorrect_data with error type
            if not strict_score:  # Only add if strict grading fails
                incorrect_data.append({
                    "example_id": i,
                    "response": response,
                    "ground_truth": gt,
                    "error_type": error_type,  # "formatting" or "math"
                    "soft_score": soft_score,  # Whether it passes soft grading
                })

        diverse.append(get_diverse_score(responses_list))

    # Use rule-based scores directly (no model-based fallback)
    final_scores = rule_based_scores

    # Calculate metrics for strict grading
    avg_scores = [sum(final_scores[i:i + num_pred]) / num_pred for i in range(0, len(final_scores), num_pred)]
    best = [max(final_scores[i:i + num_pred]) for i in range(0, len(final_scores), num_pred)]

    solve_none = sum(1 for avg_score in avg_scores if avg_score == 0)
    solve_all = sum(1 for avg_score in avg_scores if avg_score == 1)

    results["mean_score"] = sum(avg_scores) / len(avg_scores)
    results["distinct_4gram"] = sum(diverse) / len(diverse)
    results["best_score"] = sum(best) / len(best)
    results["solve_none"] = solve_none
    results["solve_all"] = solve_all
    results["avg_output_length"] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    results["format_error_rollouts"] = without_boxed

    # Calculate metrics for soft grading
    soft_avg_scores = [sum(soft_scores[i:i + num_pred]) / num_pred for i in range(0, len(soft_scores), num_pred)]
    soft_best = [max(soft_scores[i:i + num_pred]) for i in range(0, len(soft_scores), num_pred)]

    soft_solve_none = sum(1 for avg_score in soft_avg_scores if avg_score == 0)
    soft_solve_all = sum(1 for avg_score in soft_avg_scores if avg_score == 1)

    results["soft_mean_score"] = sum(soft_avg_scores) / len(soft_avg_scores)
    results["soft_best_score"] = sum(soft_best) / len(soft_best)
    results["soft_solve_none"] = soft_solve_none
    results["soft_solve_all"] = soft_solve_all

    # Save incorrect responses and ground truths to a separate file
    incorrect_file = EVAL_DIR / f"{file_path.stem}_incorrect_data.json"
    with incorrect_file.open("w", encoding="utf-8") as f:
        json.dump(incorrect_data, f, indent=4)

    return results

def main():
    all_results = []
    for file_path in EVAL_DIR.glob("*.jsonl"):
        print(f"Processing file: {file_path}")
        file_result = grade_file(file_path)
        if file_result:
            all_results.append(file_result)

    # Save results to JSON
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Grading results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

