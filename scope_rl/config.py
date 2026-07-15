"""Baseline: outcome-only GRPO (paper's control, Table 1 'GRPO' row).

Paper config (verl/recipe/scope_rl/run_qwen.sh): Qwen3-8B, GRPO group n=8,
48 prompts/step fully on-policy (mini_batch == batch), lr 1e-6 (full FT),
max_prompt 1024 / max_response 8192, no KL loss, no entropy bonus,
thinking disabled, greedy validation.

Deviations here (Tinker):
- LoRA rank 32 instead of full fine-tuning (Tinker trains LoRA only), so the
  learning rate is scaled up ~10x from the paper's full-FT value.
- Trained on the paper repo's 2400-problem DAPO-Math subset (the decomposed
  subset shipped with the repo) for a bounded number of steps, not 20k steps
  over DAPO-Math-17k.
"""

BASE_MODEL = "Qwen/Qwen3-8B"
LORA_RANK = 32

# GRPO
GROUP_SIZE = 8                # rollout.n
PROMPTS_PER_STEP = 48         # train_prompt_bsz == ppo_mini_batch_size (on-policy)
LEARNING_RATE = 1e-5          # paper: 1e-6 full FT; x10 for LoRA
ADV_NORM_STD = True           # original GRPO: (r - mean) / std
TOTAL_STEPS = 50  # one epoch over the 2400-prompt subset (2400 / 48)
ROLLOUT_TEMPERATURE = 1.0
MAX_PROMPT_TOKENS = 1024
MAX_RESPONSE_TOKENS = 8192

# Eval (paper benchmark: AIME24/25, MATH500, GPQA-diamond; greedy decoding)
EVAL_EVERY = 25               # also evals at step 0 (val_before_train) and at the end
EVAL_TEMPERATURE = 0.0
EVAL_MAX_TOKENS = 16384       # benchmark/eval.py ROLL_OUT_MAX_TOKENS
EVAL_DEDUPE = True            # AIME rows are repeated x8 for avg@8; greedy makes repeats identical

# Data lives on the HF Hub (prepared by prep_data.py), not in git.
HF_DATA_REPO = "alphaXiv/scope-rl-reproduction-data"
TRAIN_FILE = "dapo_math_2400.jsonl"
EVAL_FILE = "benchmark.jsonl"

SEED = 0
SAVE_STATE_EVERY = 10

# Resume a crashed/stalled run from a saved full state (weights + optimizer).
# Set to a tinker:// path printed as CHECKPOINT in the previous run's log, with
# RESUME_STEP = the step that checkpoint was saved at. Data order is replayed
# deterministically from SEED. None = fresh run.
RESUME_STATE_PATH = "tinker://2ef7cb7f-341f-5d1d-847e-744a30879590:train:0/weights/state-0025"
RESUME_STEP = 25
WANDB_PROJECT = "scope-rl-tinker"
RUN_NAME = "grpo-baseline"
