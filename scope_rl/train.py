"""ASR (SCOPE-RL Stage 1): adaptive scaffolded RL on Tinker (arXiv 2607.11506).

Extends the outcome-only GRPO baseline with the paper's Stage 1 (Sec. 3.2):
each step first rolls out every prompt group on the original problem and scores
it with the 0/1 outcome reward. Groups whose mean outcome reward falls below
ASR_TAU are routed: re-rolled out on the cached scaffolded prompt (sub-questions
+ main problem, answers hidden) and scored with the prefix-consistent scaffold
reward R_ASR (Eq. 10). Scaffolded rollouts REPLACE the original rollouts in the
GRPO update (compute/gradient-batch parity with the baseline, Appendix J);
non-routed groups keep their original rollouts and outcome rewards.

Claims under test vs. the baseline branch at the same budget (paper Table 1 /
Fig 3a): higher benchmark accuracy (biggest on AIME) and a higher effective
gradient ratio (fraction of groups with non-degenerate advantages).

Everything needed for analysis is printed to stdout as `METRICS {...}` /
`EVAL {...}` JSON lines. Optional W&B mirroring if WANDB_API_KEY is set.
"""

import json
import os
import random
import time

# All HF resources this run touches (Qwen tokenizer, our dataset repo) are
# public. The token orx injects into run environments can go stale (observed:
# 401 "OAuth token signature verification failed" killing the tokenizer load),
# so drop it and force anonymous Hub access before anything imports HF.
for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
    os.environ.pop(_k, None)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import numpy as np
import tinker
from huggingface_hub import hf_hub_download
from tinker import types

import config as C
from scaffold import answer_matches, asr_reward, extract_labeled_answers, load_scaffolds, prefix_len
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
                # Hard timeout so a network outage can't wedge the run forever
                # inside one blocking result() (observed: 3h JWT-refresh loop).
                results[i] = fut.result(timeout=1200)
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
    if C.RESUME_STATE_PATH:
        log(f"RESUME from {C.RESUME_STATE_PATH} at step {C.RESUME_STEP}")
        training_client = service_client.create_training_client_from_state_with_optimizer(
            C.RESUME_STATE_PATH
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=C.BASE_MODEL, rank=C.LORA_RANK
        )
    tokenizer = training_client.get_tokenizer()

    train_rows = load_hub_jsonl(C.TRAIN_FILE)
    eval_rows = load_hub_jsonl(C.EVAL_FILE)
    scaffold_rows = load_scaffolds(
        hf_hub_download(repo_id=C.HF_DATA_REPO, filename=C.SCAFFOLD_FILE, repo_type="dataset")
    )

    # Pre-tokenize training prompts; drop overlong ones (paper: filter_overlong_prompts).
    items = []
    for r in train_rows:
        toks = render_chat(tokenizer, r["messages"])
        if len(toks) <= C.MAX_PROMPT_TOKENS:
            items.append(
                {
                    "id": r["id"],
                    "toks": toks,
                    "gt": r["ground_truth"],
                    "src": r.get("data_source", "math"),
                }
            )
    # Pre-tokenize scaffolded prompts (routing targets); overlong scaffolds never route.
    scaf_items = {}
    n_scaf_overlong = 0
    for sid, s in scaffold_rows.items():
        toks = render_chat(tokenizer, s["messages"])
        if len(toks) <= C.SCAFFOLD_MAX_PROMPT_TOKENS:
            scaf_items[sid] = {"toks": toks, "sub_gts": s["sub_gts"], "gt": s["ground_truth"]}
        else:
            n_scaf_overlong += 1
    log(
        f"DATA train={len(items)} (filtered {len(train_rows) - len(items)} overlong) "
        f"scaffolds={len(scaf_items)} (filtered {n_scaf_overlong} overlong) eval={len(eval_rows)}"
    )

    rollout_params = types.SamplingParams(
        max_tokens=C.MAX_RESPONSE_TOKENS,
        temperature=C.ROLLOUT_TEMPERATURE,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    adam = types.AdamParams(learning_rate=C.LEARNING_RATE)

    def data_position(upto_step):
        """Data-order state after consuming upto_step batches, replayed from SEED.

        Uses a dedicated RNG (not the global `random` module) so replay is exact
        regardless of what other libraries draw from the global state.
        """
        r = random.Random(C.SEED)
        order = list(range(len(items)))
        r.shuffle(order)
        cursor = 0
        for _ in range(upto_step * C.PROMPTS_PER_STEP):
            if cursor >= len(order):
                r.shuffle(order)
                cursor = 0
            cursor += 1
        return r, order, cursor

    tokens_sampled = 0
    tokens_trained = 0

    start_step = C.RESUME_STEP + 1 if C.RESUME_STATE_PATH else 1
    rng, order, cursor = data_position(start_step - 1)

    sampling_client = training_client.save_weights_and_get_sampling_client()
    if C.RESUME_STATE_PATH:
        last_state_path, last_state_step = C.RESUME_STATE_PATH, C.RESUME_STEP
    else:
        # Anchor state so auto-recovery works even before the first periodic save.
        state = training_client.save_state(name="state-0000", overwrite=True).result(C.STEP_OP_TIMEOUT)
        last_state_path, last_state_step = getattr(state, "path", str(state)), 0
        log(f"CHECKPOINT step=0 path={last_state_path}")
        run_eval(sampling_client, tokenizer, eval_rows, step=0, wandb_run=wandb_run)

    step = start_step
    recoveries = 0
    while step <= C.TOTAL_STEPS:
      try:
        t0 = time.time()
        batch_idx = []
        for _ in range(C.PROMPTS_PER_STEP):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            batch_idx.append(order[cursor])
            cursor += 1
        batch = [items[i] for i in batch_idx]

        # Phase 1: original-prompt rollouts, outcome reward (identical to baseline).
        results = sample_all(
            sampling_client, [b["toks"] for b in batch], C.GROUP_SIZE, rollout_params
        )

        groups = []  # per-group dicts carrying whichever branch's rollouts enter the update
        n_outcome_effective = 0
        outcome_rewards_all = []
        for b, resp in zip(batch, results):
            if resp is None:
                continue
            seqs = resp.sequences
            rewards = []
            for seq in seqs:
                text = tokenizer.decode(list(seq.tokens), skip_special_tokens=True)
                rewards.append(score_response(text, b["gt"], b["src"]))
                tokens_sampled += len(seq.tokens)
            outcome_rewards_all.extend(rewards)
            arr = np.array(rewards, dtype=np.float64)
            if arr.std() > 0:
                n_outcome_effective += 1
            groups.append(
                {
                    "b": b,
                    "prompt_toks": b["toks"],
                    "seqs": seqs,
                    "rewards": rewards,
                    "routed": float(arr.mean()) < C.ASR_TAU and b["id"] in scaf_items,
                }
            )
        n_groups = len(groups)

        # Phase 2: routed groups re-roll out on the scaffolded prompt; scaffold
        # rollouts replace the original ones (Appendix J: replace, not augment).
        routed_groups = [g for g in groups if g["routed"]]
        prefix_lens = []
        if routed_groups:
            scaf_results = sample_all(
                sampling_client,
                [scaf_items[g["b"]["id"]]["toks"] for g in routed_groups],
                C.GROUP_SIZE,
                rollout_params,
            )
            for g, resp in zip(routed_groups, scaf_results):
                if resp is None:
                    g["routed"] = False  # sampling failed: keep original rollouts
                    continue
                sc = scaf_items[g["b"]["id"]]
                rewards = []
                for seq in resp.sequences:
                    text = tokenizer.decode(list(seq.tokens), skip_special_tokens=True)
                    sub_preds, main_pred = extract_labeled_answers(text, len(sc["sub_gts"]))
                    sub_ok = [answer_matches(p, gt) for p, gt in zip(sub_preds, sc["sub_gts"])]
                    main_ok = answer_matches(main_pred, sc["gt"])
                    rewards.append(asr_reward(sub_ok, main_ok, C.ASR_BETA))
                    prefix_lens.append(prefix_len(sub_ok) / max(len(sc["sub_gts"]), 1))
                    tokens_sampled += len(seq.tokens)
                g["prompt_toks"] = sc["toks"]
                g["seqs"] = resp.sequences
                g["rewards"] = rewards

        # GRPO update over the merged batch (identical advantage rule to baseline).
        datums = []
        rewards_all = []
        resp_lens = []
        n_effective_groups = 0
        for g in groups:
            rewards_all.extend(g["rewards"])
            for seq in g["seqs"]:
                resp_lens.append(len(seq.tokens))
            arr = np.array(g["rewards"], dtype=np.float64)
            if arr.std() == 0:
                continue  # no within-group signal -> zero advantage for every rollout
            n_effective_groups += 1
            advs = arr - arr.mean()
            if C.ADV_NORM_STD:
                advs = advs / (arr.std() + 1e-6)
            for seq, adv in zip(g["seqs"], advs):
                datums.append(make_datum(g["prompt_toks"], list(seq.tokens), list(seq.logprobs), adv))
                tokens_trained += len(seq.tokens)

        if datums:
            fb_future = training_client.forward_backward(datums, loss_fn="importance_sampling")
            opt_future = training_client.optim_step(adam)
            fb_future.result(C.STEP_OP_TIMEOUT)
            opt_future.result(C.STEP_OP_TIMEOUT)
        else:
            log(f"WARN step {step}: no effective groups, skipping update")

        sampling_client = training_client.save_weights_and_get_sampling_client()

        n_routed = sum(1 for g in groups if g["routed"])
        metrics = {
            "step": step,
            "reward_mean": round(float(np.mean(rewards_all)), 4) if rewards_all else None,
            "outcome_reward_mean": round(float(np.mean(outcome_rewards_all)), 4)
            if outcome_rewards_all
            else None,
            "routed_frac": round(n_routed / max(n_groups, 1), 4),
            "effective_group_ratio": round(n_effective_groups / max(n_groups, 1), 4),
            "effective_group_ratio_outcome": round(n_outcome_effective / max(n_groups, 1), 4),
            "scaffold_prefix_frac_mean": round(float(np.mean(prefix_lens)), 4)
            if prefix_lens
            else None,
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
            state = training_client.save_state(name=f"state-{step:04d}", overwrite=True).result(
                C.STEP_OP_TIMEOUT
            )
            last_state_path, last_state_step = getattr(state, "path", str(state)), step
            log(f"CHECKPOINT step={step} path={last_state_path}")

        if C.EVAL_EVERY and step % C.EVAL_EVERY == 0 and step != C.TOTAL_STEPS:
            run_eval(sampling_client, tokenizer, eval_rows, step=step, wandb_run=wandb_run)
      except Exception as e:  # noqa: BLE001
        recoveries += 1
        if recoveries > C.MAX_RECOVERIES:
            log(f"FATAL recovery budget exhausted at step {step}: {e}")
            raise
        log(
            "RECOVER "
            + json.dumps(
                {
                    "failed_step": step,
                    "resume_from_step": last_state_step,
                    "state": last_state_path,
                    "recoveries": recoveries,
                    "error": str(e)[:300] or type(e).__name__,
                }
            )
        )
        training_client = service_client.create_training_client_from_state_with_optimizer(
            last_state_path
        )
        sampling_client = training_client.save_weights_and_get_sampling_client()
        rng, order, cursor = data_position(last_state_step)
        step = last_state_step + 1
        continue
      step += 1

    state = training_client.save_state(name="state-final", overwrite=True).result(C.STEP_OP_TIMEOUT)
    log(f"CHECKPOINT final path={getattr(state, 'path', state)}")
    run_eval(sampling_client, tokenizer, eval_rows, step=C.TOTAL_STEPS, wandb_run=wandb_run)
    log("DONE")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
