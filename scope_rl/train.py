"""Fig 1a probe: scaffold-prefix progress vs. main-answer accuracy (no training).

Reproduces the paper's motivating probe (Appendix C) on the base model: the 755
training problems whose scaffold decomposition has exactly 4 sub-questions are
evaluated under (a) the scaffolded prompt (sub-questions + main problem in one
generation) and (b) the original prompt. Samples are binned by the longest
correct sub-question prefix from the scaffold rollout; per bin we report
main-answer accuracy under both prompts, verified against the same ground truth.

Claim under test (paper Fig 1a): scaffold-prompt accuracy exceeds original-prompt
accuracy and rises monotonically with prefix length
(paper: 25.4/36.2/60.3/90.4% vs 29.6/31.9/39.7/49.3% for bins 1-4).

No parameter updates happen; the LoRA client is created only to obtain a
tokenizer and a base-model sampling client (LoRA is identity at init).
Results are printed as `PROBE ...` / `PROBE_RESULT ...` JSON lines.
"""

import json

import numpy as np
import tinker
from huggingface_hub import hf_hub_download
from tinker import types

import config as C
from scaffold import answer_matches, extract_labeled_answers, load_scaffolds, prefix_len
from verifier import score_response


def log(msg: str) -> None:
    print(msg, flush=True)


def load_hub_jsonl(filename):
    path = hf_hub_download(repo_id=C.HF_DATA_REPO, filename=filename, repo_type="dataset")
    with open(path) as f:
        return [json.loads(line) for line in f]


def render_chat(tokenizer, messages):
    """Tokenize a chat prompt with the generation header, thinking disabled."""
    try:
        out = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def sample_all(sampling_client, prompts_tokens, num_samples, sampling_params, max_retries=3):
    """Submit one sample request per prompt, gather results in order."""
    results = [None] * len(prompts_tokens)
    pending = list(range(len(prompts_tokens)))
    for attempt in range(max_retries):
        futures = {}
        for i in pending:
            futures[i] = sampling_client.sample(
                prompt=types.ModelInput.from_ints(prompts_tokens[i]),
                num_samples=num_samples,
                sampling_params=sampling_params,
            )
        failed = []
        for i, fut in futures.items():
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append(i)
                if attempt == max_retries - 1:
                    log(f"WARN sample failed permanently for prompt {i}: {e}")
        pending = failed
        if not pending:
            break
    return results


def main() -> None:
    log(f"CONFIG {json.dumps({k: str(v) for k, v in vars(C).items() if k.isupper()})}")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=C.BASE_MODEL, rank=C.LORA_RANK
    )
    tokenizer = training_client.get_tokenizer()
    sampling_client = training_client.save_weights_and_get_sampling_client()

    scaffolds = load_scaffolds(hf_hub_download(
        repo_id=C.HF_DATA_REPO, filename=C.SCAFFOLD_FILE, repo_type="dataset"
    ))
    originals = {r["id"]: r for r in load_hub_jsonl(C.TRAIN_FILE)}
    rows = [
        s
        for s in scaffolds.values()
        if len(s["sub_gts"]) == C.PROBE_N_SUBS and s["id"] in originals
    ]
    log(f"DATA probe problems with exactly {C.PROBE_N_SUBS} subs: {len(rows)}")

    params = types.SamplingParams(
        max_tokens=C.PROBE_MAX_TOKENS,
        temperature=C.PROBE_TEMPERATURE,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )

    # Condition (a): scaffolded prompt.
    scaf_prompts = [render_chat(tokenizer, s["messages"]) for s in rows]
    # Condition (b): original prompt (same problems, same ground truth).
    orig_prompts = [render_chat(tokenizer, originals[s["id"]]["messages"]) for s in rows]

    log(f"SAMPLING scaffold condition ({len(rows)} prompts)")
    scaf_results = sample_all(sampling_client, scaf_prompts, 1, params)
    log(f"SAMPLING original condition ({len(rows)} prompts)")
    orig_results = sample_all(sampling_client, orig_prompts, 1, params)

    records = []
    n_missing = 0
    for s, scaf_resp, orig_resp in zip(rows, scaf_results, orig_results):
        if scaf_resp is None or orig_resp is None:
            n_missing += 1
            continue
        scaf_text = tokenizer.decode(list(scaf_resp.sequences[0].tokens), skip_special_tokens=True)
        orig_text = tokenizer.decode(list(orig_resp.sequences[0].tokens), skip_special_tokens=True)
        sub_preds, main_pred = extract_labeled_answers(scaf_text, len(s["sub_gts"]))
        sub_ok = [answer_matches(p, gt) for p, gt in zip(sub_preds, s["sub_gts"])]
        records.append(
            {
                "id": s["id"],
                "prefix_len": prefix_len(sub_ok),
                "scaf_main_ok": answer_matches(main_pred, s["ground_truth"]),
                "orig_main_ok": score_response(orig_text, s["ground_truth"], "math") > 0,
                "scaf_tokens": len(scaf_resp.sequences[0].tokens),
                "orig_tokens": len(orig_resp.sequences[0].tokens),
            }
        )
    if n_missing:
        log(f"WARN {n_missing} problems dropped due to sampling failures")

    bins = {}
    for r in records:
        bins.setdefault(r["prefix_len"], []).append(r)

    def bin_stats(rs):
        return {
            "n": len(rs),
            "scaffold_main_acc": round(float(np.mean([r["scaf_main_ok"] for r in rs])), 4),
            "original_main_acc": round(float(np.mean([r["orig_main_ok"] for r in rs])), 4),
        }

    for L in sorted(bins):
        log("PROBE " + json.dumps({"prefix_len": L, **bin_stats(bins[L])}))

    overall = {
        **bin_stats(records),
        "mean_prefix_len": round(float(np.mean([r["prefix_len"] for r in records])), 3),
        "scaf_tokens_mean": round(float(np.mean([r["scaf_tokens"] for r in records])), 1),
        "orig_tokens_mean": round(float(np.mean([r["orig_tokens"] for r in records])), 1),
        "bins": {str(L): bin_stats(bins[L]) for L in sorted(bins)},
    }
    log("PROBE_RESULT " + json.dumps(overall))
    log("DONE")


if __name__ == "__main__":
    main()
