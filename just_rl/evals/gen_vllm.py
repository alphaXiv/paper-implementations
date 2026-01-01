import os
import json
import random
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# --------------------------------------------------------------------------- #
#                   Global constants / variables                              #
# --------------------------------------------------------------------------- #
DATA_DIR = "data"
TASKS       = [
    {"name": "AIME24", "path": f"{DATA_DIR}/AIME24/test.parquet", "N": 32},
    {"name": "AIME25", "path": f"{DATA_DIR}/AIME25/test.parquet", "N": 32},
    {"name": "AMC23", "path": f"{DATA_DIR}/AMC23/test.parquet", "N": 32},
    {"name": "MATH-500", "path": f"{DATA_DIR}/MATH-500/test.parquet", "N": 4},
    {"name": "Minerva", "path": f"{DATA_DIR}/Minerva/test.parquet", "N": 4},
    {"name": "Olympiad-Bench", "path": f"{DATA_DIR}/Olympiad-Bench/test.parquet", "N": 4},
    {"name": "BRUMO25", "path": f"{DATA_DIR}/BRUMO25/test.parquet", "N": 32},
    {"name": "CMIMC25", "path": f"{DATA_DIR}/CMIMC25/test.parquet", "N": 32},
    {"name": "HMMT25", "path": f"{DATA_DIR}/HMMT25/test.parquet", "N": 32},
]
PROMPT_TEMPLATE = """{problem} Please reason step by step, and put your final answer within \\boxed{{}}."""
NAME        = "HuggingFaceTB/SmolLM-360M"
MAX_TOKENS  = 31744
TEMPERATURE = 0.7
TOP_P       = 0.9
OUT_DIR     = Path(f"justrl_eval_outputs/{NAME.split('/')[-1]}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
    """Read parquet file and return a list of prompts (no duplication)."""
    df = pd.read_parquet(filepath)
    if "BRUMO25" in filepath or "CMIMC25" in filepath or "HMMT25" in filepath:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "problem"].strip(),
                "answer": df.at[i, "answer"].strip(),
            }
            for i in range(len(df))
        ]
    else:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "prompt"][0]["content"].strip(),
                "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
            }
            for i in range(len(df))
        ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def split_seeds(seeds: list[int], num_workers: int):
    """Round-robin split of the seed list into num_workers chunks."""
    chunks = [[] for _ in range(num_workers)]
    for idx, s in enumerate(seeds):
        chunks[idx % num_workers].append(s)
    return chunks


# --------------------------------------------------------------------------- #
#                           Worker process (one GPU)                          #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
    """
    Each worker runs on a single GPU:

    args_tuple = (samples, seed_list, gpu_id)
    """
    samples, seed_list, gpu_id = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] seeds={seed_list} | loading model...", flush=True)

    llm = LLM(model=NAME, enforce_eager=True)
    results = []

    for seed in seed_list:
        sampling = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=seed,
        )
        # Changed from the following code because of this error:
        # ValueError: As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if the tokenizer does not define one.
        #  
        # messages = [[{"role": "user", "content": s["prompt"]}] for s in samples]
        # outputs = llm.chat(messages, sampling, use_tqdm=True)


        prompts = [s["prompt"] for s in samples]
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        for sample, out in zip(samples, outputs):
            results.append(
                {
                    "example_id": sample["example_id"],
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "seed": seed,
                    "response": out.outputs[0].text,
                }
            )
    return results


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    available_workers = [0,1,2,3,4,5,6,7]
    num_workers = len(available_workers)
    for task in TASKS:
        task_name = task["name"]
        task_path = task["path"]
        N = task["N"]

        print(f"Starting evaluation for task: {task_name} (N={N})")

        # Update output path for the current task
        out_path = OUT_DIR / f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{N}-MNT{MAX_TOKENS}.jsonl"

        # 1. Load original prompts
        samples = load_samples(task_path)

        # Append suffix prompt to each sample
        for sample in samples:
            sample["prompt"] = PROMPT_TEMPLATE.format(problem=sample["prompt"])

        # demo print
        print("Example prompt after formatting:")
        print(samples[0]["prompt"])
        
        # 2. Generate N distinct random seeds and split across GPUs
        random_seeds = random.sample(range(2**31 - 1), N)  # unique & shuffled
        seed_chunks = split_seeds(random_seeds, num_workers)

        # 3. Launch workers
        all_results = []
        args_list = [(samples, seed_chunks[i], gid) for (i, gid) in enumerate(available_workers)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(worker_process, tup) for tup in args_list]
            for fut in tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures), desc=f"GPU workers ({task_name})"):
                all_results.extend(fut.result())

        print(f"Total generations collected for {task_name}: {len(all_results)}")  # len(samples) * N

        # 4. Save to disk
        with out_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved results for {task_name} to {out_path}")


if __name__ == "__main__":
    main()