"""ASR (SCOPE-RL Stage 1): adaptive scaffolded RL (paper Sec. 3.2, Table 1 'ASR only' row).

Identical optimizer/budget to the baseline branch (LoRA r32, lr 1e-5, 50 steps,
group 8, 48 prompts/step); the only change is the reward path: groups with mean
outcome reward < ASR_TAU are re-rolled out on the paper's cached scaffolded
prompts and scored with the prefix-consistent scaffold reward (Eq. 10).

Paper values kept: tau=0.5, beta=0.5, G=8 (Sec. 3.2.2-3.2.3).
Deviations (same as baseline): LoRA instead of full FT, 50 steps, 2400-problem
subset. ASR-specific deviation: scaffolded prompts are capped at 2048 tokens
(paper caps original prompts at 1024 and does not state a scaffold cap);
overlong scaffolds simply never route.
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

# ASR (Stage 1)
ASR_TAU = 0.5                 # routing threshold on group mean outcome reward (Eq. 8)
ASR_BETA = 0.5                # scaffold-checkpoint weight in R_ASR (Eq. 10)
SCAFFOLD_MAX_PROMPT_TOKENS = 2048

# Eval (paper benchmark: AIME24/25, MATH500, GPQA-diamond; greedy decoding)
EVAL_EVERY = 25               # also evals at step 0 (val_before_train) and at the end
EVAL_TEMPERATURE = 0.0
EVAL_MAX_TOKENS = 16384       # benchmark/eval.py ROLL_OUT_MAX_TOKENS
EVAL_DEDUPE = True            # AIME rows are repeated x8 for avg@8; greedy makes repeats identical

# Data lives on the HF Hub (prepared by prep_data.py), not in git.
HF_DATA_REPO = "alphaXiv/scope-rl-reproduction-data"
TRAIN_FILE = "dapo_math_2400.jsonl"
SCAFFOLD_FILE = "dapo_math_2400_scaffold.jsonl"
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
RUN_NAME = "asr-stage1"
