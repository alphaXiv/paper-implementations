"""Prepare training/eval data from the paper's repo (github.com/tokencraft-lab/SCOPE-RL).

Training rows keep only what outcome-only GRPO needs: the original chat prompt and
the rule-verifiable ground truth. The embedded sub-question decompositions (used by
the ASR stage) are dropped here; the ASR experiment branch re-extracts them.

Usage:
    python prep_data.py --scope-rl-repo /path/to/SCOPE-RL [--train-source dapo-math-2400]
"""

import argparse
import json
from pathlib import Path

TRAIN_SOURCES = {
    "dapo-math-2400": "data/dapo-math-2400-embedded.jsonl",
    "dapo-math-17k": "data/dapo-math-17k-embedded.jsonl",
    "big-math-2400": "data/big-math-2400-embedded.jsonl",
    "big-math-12k": "data/big-math-12k-embedded.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope-rl-repo", required=True)
    ap.add_argument("--train-source", default="dapo-math-2400", choices=TRAIN_SOURCES)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    repo = Path(args.scope_rl_repo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_in, n_out = 0, 0
    train_out = out_dir / f"{args.train_source.replace('-', '_')}.jsonl"
    with open(repo / TRAIN_SOURCES[args.train_source]) as f, open(train_out, "w") as w:
        for line in f:
            n_in += 1
            row = json.loads(line)
            gt = row["reward_model"]["ground_truth"]
            if not isinstance(gt, str) or not row.get("prompt"):
                continue
            w.write(
                json.dumps(
                    {
                        "id": row.get("extra_info", {}).get("index", str(n_in)),
                        "data_source": row.get("data_source", "math_dapo"),
                        "messages": row["prompt"],
                        "ground_truth": gt,
                    }
                )
                + "\n"
            )
            n_out += 1
    print(f"train: {n_in} rows read -> {n_out} written to {train_out}")

    bench_in = repo / "benchmark/data/data.jsonl"
    bench_out = out_dir / "benchmark.jsonl"
    n = 0
    with open(bench_in) as f, open(bench_out, "w") as w:
        for line in f:
            w.write(line)
            n += 1
    print(f"benchmark: {n} rows copied to {bench_out}")


if __name__ == "__main__":
    main()
