"""Prepare training/eval data from the paper's repo (github.com/tokencraft-lab/SCOPE-RL).

Training rows keep only what outcome-only GRPO needs: the original chat prompt and
the rule-verifiable ground truth. The embedded sub-question decompositions (used by
the ASR stage) are dropped here; the ASR experiment branch re-extracts them.

The prepared files are not committed to git; they are pushed to the HF Hub
dataset repo that train.py downloads from (config.HF_DATA_REPO).

Usage:
    python prep_data.py --scope-rl-repo /path/to/SCOPE-RL [--train-source dapo-math-2400] [--push-to-hub]
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
    ap.add_argument("--push-to-hub", action="store_true", help="upload prepared files to config.HF_DATA_REPO")
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

    # Scaffold chains (verifier-only sub-answers) for the ASR stage / Fig 1a probe.
    n_in, n_scaf = 0, 0
    scaffold_out = out_dir / f"{args.train_source.replace('-', '_')}_scaffold.jsonl"
    with open(repo / TRAIN_SOURCES[args.train_source]) as f, open(scaffold_out, "w") as w:
        for line in f:
            n_in += 1
            row = json.loads(line)
            main_gt = row["reward_model"]["ground_truth"]
            dec = (row.get("extra_info") or {}).get("decomposition_result") or {}
            gt = (dec.get("reward_model") or {}).get("ground_truth")
            if not dec.get("prompt") or not isinstance(gt, dict) or not isinstance(main_gt, str):
                continue
            subs = sorted((k for k in gt if k.startswith("sub")), key=lambda k: int(k[3:]))
            if not subs or "main" not in gt:
                continue
            w.write(
                json.dumps(
                    {
                        "id": row.get("extra_info", {}).get("index", str(n_in)),
                        "messages": dec["prompt"],
                        "sub_gts": [str(gt[k]) for k in subs],
                        "ground_truth": main_gt,
                    }
                )
                + "\n"
            )
            n_scaf += 1
    print(f"scaffold: {n_in} rows read -> {n_scaf} written to {scaffold_out}")

    bench_in = repo / "benchmark/data/data.jsonl"
    bench_out = out_dir / "benchmark.jsonl"
    n = 0
    with open(bench_in) as f, open(bench_out, "w") as w:
        for line in f:
            w.write(line)
            n += 1
    print(f"benchmark: {n} rows copied to {bench_out}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        import config

        api = HfApi()
        api.create_repo(config.HF_DATA_REPO, repo_type="dataset", exist_ok=True)
        for path in [train_out, scaffold_out, bench_out]:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=config.HF_DATA_REPO,
                repo_type="dataset",
            )
            print(f"pushed {path.name} to hf.co/datasets/{config.HF_DATA_REPO}")


if __name__ == "__main__":
    main()
