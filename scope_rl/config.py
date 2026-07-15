"""Fig 1a probe: scaffold prefix vs. main-answer accuracy (base model, no RL).

Paper setup (Appendix C): 755 training problems whose scaffold has exactly 4
sub-questions, evaluated on the base model under scaffolded vs. original
prompts, binned by longest correct sub-question prefix.

Deviations here:
- Greedy decoding (temp 0) instead of the paper's unstated sampling setup, so
  the prefix-length binning is deterministic.
- Same Tinker sampling path as the training experiments (LoRA client at init ==
  base model), keeping the serving stack constant across the tree.
"""

BASE_MODEL = "Qwen/Qwen3-8B"
LORA_RANK = 32

# Probe
PROBE_N_SUBS = 4              # paper: exactly-4-sub-question problems (755 of 2,382)
PROBE_TEMPERATURE = 0.0
PROBE_MAX_TOKENS = 16384      # scaffold rollouts answer 4 subs + main in one generation

# Data lives on the HF Hub (prepared by prep_data.py), not in git.
HF_DATA_REPO = "alphaXiv/scope-rl-reproduction-data"
TRAIN_FILE = "dapo_math_2400.jsonl"
SCAFFOLD_FILE = "dapo_math_2400_scaffold.jsonl"

SEED = 0
