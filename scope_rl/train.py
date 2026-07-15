"""Outcome-only GRPO baseline on Tinker (SCOPE-RL reproduction, arXiv 2607.11506).

One training step = sample GROUP_SIZE rollouts for each of PROMPTS_PER_STEP
prompts from the current policy, reward each rollout 1/0 by rule-based final-
answer check, form group-relative advantages, and take one importance-sampling
policy-gradient step (fully on-policy, matching the paper's mini_batch == batch).

Everything needed for analysis is printed to stdout as `METRICS {...}` /
`EVAL {...}` JSON lines. Optional W&B mirroring if WANDB_API_KEY is set.
"""

import json
import os
import random
import time

import numpy as np
import tinker
from tinker import types

import config as C
from verifier import score_response


def log(msg: str) -> None:
    print(msg, flush=True)


def load_jsonl(path):
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
    if hasattr(out, "keys"):  # transformers >= 5 returns a BatchEncoding
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


def make_datum(prompt_toks, seq_tokens, seq_logprobs, advantage):
    full = list(prompt_toks) + list(seq_tokens)
    input_toks = full[:-1]
    targets = full[1:]
    n_prompt = len(prompt_toks)
    logprobs = [0.0] * (n_prompt - 1) + list(seq_logprobs)
    advs = [0.0] * (n_prompt - 1) + [float(advantage)] * len(seq_tokens)
    assert len(input_toks) == len(targets) == len(logprobs) == len(advs)
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_toks),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_numpy(np.array(targets, dtype=np.int64)),
            "logprobs": types.TensorData.from_numpy(np.array(logprobs, dtype=np.float32)),
            "advantages": types.TensorData.from_numpy(np.array(advs, dtype=np.float32)),
        },
    )


def run_eval(sampling_client, tokenizer, eval_rows, step, wandb_run=None):
    """Greedy decode on the paper benchmark; rule-based verify; truncated => wrong."""
    rows = eval_rows
    if C.EVAL_DEDUPE:
        seen, rows = set(), []
        for r in eval_rows:
            key = r["problem"]
            if key not in seen:
                seen.add(key)
                rows.append(r)
    prompts = [render_chat(tokenizer, [{"role": "user", "content": r["problem"]}]) for r in rows]
    params = types.SamplingParams(
        max_tokens=C.EVAL_MAX_TOKENS,
        temperature=C.EVAL_TEMPERATURE,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    t0 = time.time()
    results = sample_all(sampling_client, prompts, 1, params)
    per_source = {}
    n_trunc = 0
    for r, resp in zip(rows, results):
        src = r["source"]
        stats = per_source.setdefault(src, {"n": 0, "correct": 0, "tokens": 0})
        stats["n"] += 1
        if resp is None:
            continue
        seq = resp.sequences[0]
        text = tokenizer.decode(list(seq.tokens), skip_special_tokens=True)
        truncated = str(getattr(seq, "stop_reason", "")) == "length" or len(seq.tokens) >= C.EVAL_MAX_TOKENS
        n_trunc += int(truncated)
        correct = (not truncated) and score_response(text, str(r["answer"]), src) > 0
        stats["correct"] += int(correct)
        stats["tokens"] += len(seq.tokens)
    summary = {"step": step, "eval_seconds": round(time.time() - t0, 1), "truncated": n_trunc}
    accs = []
    for src, s in sorted(per_source.items()):
        acc = s["correct"] / max(s["n"], 1)
        accs.append(acc)
        summary[f"acc/{src}"] = round(acc, 4)
        summary[f"avg_tokens/{src}"] = round(s["tokens"] / max(s["n"], 1), 1)
    summary["acc/avg"] = round(float(np.mean(accs)), 4) if accs else 0.0
    log("EVAL " + json.dumps(summary))
    if wandb_run:
        wandb_run.log({f"eval/{k}": v for k, v in summary.items() if k != "step"}, step=step)
    return summary


def main() -> None:
    random.seed(C.SEED)
    np.random.seed(C.SEED)

    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=C.WANDB_PROJECT, name=C.RUN_NAME, config={k: v for k, v in vars(C).items() if k.isupper()}
            )
        except Exception as e:  # noqa: BLE001
            log(f"WARN wandb init failed, continuing without: {e}")

    log(f"CONFIG {json.dumps({k: v for k, v in vars(C).items() if k.isupper()})}")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=C.BASE_MODEL, rank=C.LORA_RANK
    )
    tokenizer = training_client.get_tokenizer()

    train_rows = load_jsonl(C.TRAIN_FILE)
    eval_rows = load_jsonl(C.EVAL_FILE)

    # Pre-tokenize training prompts; drop overlong ones (paper: filter_overlong_prompts).
    items = []
    for r in train_rows:
        toks = render_chat(tokenizer, r["messages"])
        if len(toks) <= C.MAX_PROMPT_TOKENS:
            items.append({"toks": toks, "gt": r["ground_truth"], "src": r.get("data_source", "math")})
    log(f"DATA train={len(items)} (filtered {len(train_rows) - len(items)} overlong) eval={len(eval_rows)}")

    rollout_params = types.SamplingParams(
        max_tokens=C.MAX_RESPONSE_TOKENS,
        temperature=C.ROLLOUT_TEMPERATURE,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    adam = types.AdamParams(learning_rate=C.LEARNING_RATE)

    order = list(range(len(items)))
    random.shuffle(order)
    cursor = 0
    tokens_sampled = 0
    tokens_trained = 0

    sampling_client = training_client.save_weights_and_get_sampling_client(name="step-0000")
    run_eval(sampling_client, tokenizer, eval_rows, step=0, wandb_run=wandb_run)

    for step in range(1, C.TOTAL_STEPS + 1):
        t0 = time.time()
        batch_idx = []
        for _ in range(C.PROMPTS_PER_STEP):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            batch_idx.append(order[cursor])
            cursor += 1
        batch = [items[i] for i in batch_idx]

        results = sample_all(
            sampling_client, [b["toks"] for b in batch], C.GROUP_SIZE, rollout_params
        )

        datums = []
        rewards_all = []
        resp_lens = []
        n_effective_groups = 0
        n_groups = 0
        for b, resp in zip(batch, results):
            if resp is None:
                continue
            n_groups += 1
            seqs = resp.sequences
            rewards = []
            for seq in seqs:
                text = tokenizer.decode(list(seq.tokens), skip_special_tokens=True)
                rewards.append(score_response(text, b["gt"], b["src"]))
                resp_lens.append(len(seq.tokens))
                tokens_sampled += len(seq.tokens)
            rewards_all.extend(rewards)
            arr = np.array(rewards, dtype=np.float64)
            if arr.std() == 0:
                continue  # no within-group signal -> zero advantage for every rollout
            n_effective_groups += 1
            advs = arr - arr.mean()
            if C.ADV_NORM_STD:
                advs = advs / (arr.std() + 1e-6)
            for seq, adv in zip(seqs, advs):
                datums.append(make_datum(b["toks"], list(seq.tokens), list(seq.logprobs), adv))
                tokens_trained += len(seq.tokens)

        if datums:
            fb_future = training_client.forward_backward(datums, loss_fn="importance_sampling")
            opt_future = training_client.optim_step(adam)
            fb_future.result()
            opt_future.result()
        else:
            log(f"WARN step {step}: no effective groups, skipping update")

        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"step-{step:04d}"
        )

        metrics = {
            "step": step,
            "reward_mean": round(float(np.mean(rewards_all)), 4) if rewards_all else None,
            "effective_group_ratio": round(n_effective_groups / max(n_groups, 1), 4),
            "n_datums": len(datums),
            "resp_len_mean": round(float(np.mean(resp_lens)), 1) if resp_lens else None,
            "tokens_sampled_total": tokens_sampled,
            "tokens_trained_total": tokens_trained,
            "step_seconds": round(time.time() - t0, 1),
        }
        log("METRICS " + json.dumps(metrics))
        if wandb_run:
            wandb_run.log({f"train/{k}": v for k, v in metrics.items() if k != "step" and v is not None}, step=step)

        if C.SAVE_STATE_EVERY and step % C.SAVE_STATE_EVERY == 0:
            state = training_client.save_state(name=f"state-{step:04d}").result()
            log(f"CHECKPOINT step={step} path={getattr(state, 'path', state)}")

        if C.EVAL_EVERY and step % C.EVAL_EVERY == 0 and step != C.TOTAL_STEPS:
            run_eval(sampling_client, tokenizer, eval_rows, step=step, wandb_run=wandb_run)

    state = training_client.save_state(name="state-final").result()
    log(f"CHECKPOINT final path={getattr(state, 'path', state)}")
    run_eval(sampling_client, tokenizer, eval_rows, step=C.TOTAL_STEPS, wandb_run=wandb_run)
    log("DONE")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
