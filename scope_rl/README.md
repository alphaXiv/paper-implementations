# SCOPE-RL: Optimizing Reasoning Paths Before and After Success

Reproduction of **SCOPE-RL** (arXiv 2607.11506, Liu & Xu et al., Baidu Inc. / Shandong
University; code: [tokencraft-lab/SCOPE-RL](https://github.com/tokencraft-lab/SCOPE-RL)).

The paper densifies the sparse verifiable reward in RLVR in two stages on top of GRPO:
**ASR** (adaptive scaffolded RL: verifiable sub-question rewards on hard problems) and
**QPR** (quality-aware process RL: correctness-gated step-quality shaping). Headline claim
(Table 1): up to **+11.2pp average accuracy** over outcome-only GRPO on AIME24/25,
MATH500, GPQA with Qwen3-8B, plus 16-27% fewer reasoning tokens.

## Reproduction setup

All RL training runs on [Tinker](https://tinker.thinkingmachines.ai) (LoRA training API);
the driver loop runs as an orx experiment. Deliberate deviations from the paper, chosen
to fit the compute budget:

| Paper | Here | Why |
|---|---|---|
| Full fine-tuning, lr 1e-6 | LoRA rank 32, lr 1e-5 | Tinker trains LoRA only; LoRA needs ~10x FT lr |
| DAPO-Math-17k, 20k steps | repo's 2400-problem DAPO subset, ~100 steps | budget |
| LLM-judge answer correctness in eval | rule-based verifier (paper's own `math_dapo.py` port) | objective + cheap |
| AIME avg@8 sampling in benchmark file | deduplicated, greedy (temp 0) | greedy repeats are identical |

Faithful to the paper: GRPO with group size 8, 48 prompts/step fully on-policy,
group-relative advantages normalized by std, 0/1 outcome reward from the paper's own
verifier, no KL/entropy terms, thinking disabled, max response 8192, greedy eval on the
paper's benchmark file.

## Files

- `train.py` — on this branch: the Stage 1 ASR loop (outcome rollout → route hard groups to scaffolded re-rollout → prefix-consistent reward → GRPO update), with periodic benchmark evals printed as `EVAL {...}` JSON lines.
- `scaffold.py` — scaffold loading, `[SUB-X ANSWER]`/`[MAIN ANSWER]` extraction, prefix-consistency (Eq. 9) and the ASR reward (Eq. 10).
- `verifier.py` — port of the paper's `math_dapo.py` Minerva-style verifier + GPQA letter matching.
- `config.py` — all hyperparameters (the experiment branch is the config).
- `prep_data.py` — regenerates the data from a clone of the paper's repo and pushes it to the Hub (`--push-to-hub`).

Data is not stored in git: `train.py` downloads it from
[`alphaXiv/scope-rl-reproduction-data`](https://huggingface.co/datasets/alphaXiv/scope-rl-reproduction-data)
(`dapo_math_2400.jsonl` — training prompts + ground truths with decompositions stripped;
`benchmark.jsonl` — the paper's eval set: AIME24/25, MATH500, GPQA-diamond).

Run: `uv sync && uv run python train.py` (needs `TINKER_API_KEY`).

## Experiments

| Experiment | Description | Branch |
|---|---|---|
| Baseline: outcome-only GRPO | Paper's control (Table 1 "GRPO" row) on DAPO-Math subset via Tinker | [`orx/scope-rl-baseline-outcome-only-grpo-tinker`](https://github.com/alphaXiv/paper-implementations/tree/orx/scope-rl-baseline-outcome-only-grpo-tinker) |
| Fig 1a probe: scaffold prefix vs main accuracy | Base model, no RL: 755 four-sub-question problems under scaffolded vs original prompt, binned by correct prefix length | [`orx/probe-fig-1a-scaffold-prefix-vs-main-answer-accu`](https://github.com/alphaXiv/paper-implementations/tree/orx/probe-fig-1a-scaffold-prefix-vs-main-answer-accu) |
| ASR (Stage 1): adaptive scaffolded RL | Paper Table 1 "ASR only" row: on-policy routing (tau=0.5) to prefix-consistent scaffold rewards (beta=0.5), same budget as baseline | [`orx/asr-stage-1-adaptive-scaffolded-rl`](https://github.com/alphaXiv/paper-implementations/tree/orx/asr-stage-1-adaptive-scaffolded-rl) |
| ASR (Stage 1) retry: crash-hardened | Same ASR experiment, completed after two network-crash losses: adds full-state resume, sample/step timeouts, 10-step checkpoints, in-process auto-recovery | [`orx/asr-stage-1-retry-crash-hardened`](https://github.com/alphaXiv/paper-implementations/tree/orx/asr-stage-1-retry-crash-hardened) |
