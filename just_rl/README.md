# JustRL Reproduction: Scaling a 1.5B LLM with a Simple RL Recipe

This repository contains a complete reproduction of the "JustRL" method described in the paper **"JustRL: Scaling a 1.5B LLM with a Simple RL Recipe"**.

The core contribution of this work is demonstrating that a simple, single-stage Group Relative Policy Optimization (GRPO) training loop with fixed hyperparameters and a basic rule-based verifier outperforms complex multi-stage RL pipelines for reasoning models.

## 📂 Directory Structure

```
justrl_reproduction/
├── config/                  # Hydra configuration files
│   ├── justrl_deepseek_1.5b.yaml
│   └── justrl_nemotron_1.5b.yaml
├── data/                    # Data loading and processing
│   ├── dapo_loader.py       # DAPO-Math-17k dataset loader
│   └── prompt_utils.py      # Prompt formatting logic
├── scripts/                 # Executable scripts
│   ├── prepare_data.sh      # Data download and preparation script
│   ├── train_grpo.py        # Main GRPO training loop
│   └── evaluate.py          # Evaluation script (AIME/MATH-500)
├── src/                     # Core implementation
│   ├── grpo_trainer.py      # Custom GRPO Trainer (Actor-only)
│   └── utils.py             # Advantage calculation helpers
├── verifiers/               # Reward functions
│   └── math_rule_verifier.py # Strict regex-based binary verifier
├── pyproject.toml           # Project dependencies and configuration
└── uv.lock                  # Locked dependency versions
```

## 🛠️ Installation

### Prerequisites
- **Python**: 3.10+
- **CUDA**: 12.1+
- **GPU**: Minimum 8x A100 (80GB) or H100 recommended (due to 16k context length).

### Setup
1. Clone the repository (if applicable) or navigate to the root directory.
2. Install [uv](https://github.com/astral-sh/uv) if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Install dependencies:

```bash
uv sync
```

*Note: This installs `vllm`, `verl`, `flash-attn`, and other core libraries required for efficient rollout generation and training. The project uses `uv` for fast, reliable dependency management.*

## 📊 Data Preparation

The method uses the **DAPO-Math-17k** dataset. We provide a script to download and format it automatically.

```bash
cd scripts
bash prepare_data.sh
```

This will:
1. Download `dapo-ai/DAPO-Math-17k` from HuggingFace.
2. Format prompts with the required suffix: `"\nPlease reason step by step, and put your final answer within \\boxed{}."`.
3. Save the processed data to `data/processed/dapo_math_17k_processed.parquet`.

## 🚀 Training

The training uses **GRPO (Group Relative Policy Optimization)** with a specific "Simple Recipe":
- **Algorithm**: GRPO (Actor-only, no Critic).
- **KL Coefficient**: 0.0 (Strictly enforced).
- **Entropy Coefficient**: 0.0.
- **Clipping**: Asymmetric `[0.8, 1.28]`.
- **Learning Rate**: `1e-6` (Constant, no decay).
- **Response Length**: Up to 15,360 tokens.

### Run Training (DeepSeek-1.5B Base)

```bash
# Run with torchrun for distributed training (example for 8 GPUs)
torchrun --nproc_per_node=8 scripts/train_grpo.py \
    config_name=justrl_deepseek_1.5b \
    data.train_files=../data/processed/dapo_math_17k_processed.parquet
```

### Run Training (Nemotron-1.5B Base)

```bash
torchrun --nproc_per_node=8 scripts/train_grpo.py \
    config_name=justrl_nemotron_1.5b \
    data.train_files=../data/processed/dapo_math_17k_processed.parquet
```

**Key Monitoring Metrics (WandB):**
- `response_length`: Should start high (~7k-8k) and naturally converge to ~4k-5k.
- `reward/mean`: Should monotonically increase.
- `entropy`: Should oscillate between 1.2 and 1.4.

## 📈 Evaluation

Evaluate the trained model on **MATH-500** or **AIME 2024** using Pass@1 accuracy.

```bash
# Evaluate on MATH-500
python scripts/evaluate.py \
    --model_path /path/to/saved/checkpoint \
    --dataset math500 \
    --tensor_parallel_size 1 \
    --output_file results_math500.jsonl

# Evaluate on AIME 2024
python scripts/evaluate.py \
    --model_path /path/to/saved/checkpoint \
    --dataset aime2024 \
    --tensor_parallel_size 1 \
    --output_file results_aime.jsonl
```

## 🧩 The "Simple Recipe" Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Learning Rate** | `1e-6` | Constant schedule. Do not decay. |
| **Global Batch Size** | 256 | Achieved via Gradient Accumulation. |
| **Group Size (G)** | 8 | Number of rollouts per prompt. |
| **Max Response Length** | 15,360 | Crucial for allowing long reasoning chains. |
| **Verifier** | Rule-based | Strict string match of `\boxed{...}` content. |
| **KL Penalty** | 0.0 | Constraints hurt reasoning performance. |

## ⚠️ Troubleshooting

- **OOM Errors**: The 16k context length is memory intensive.
  - Reduce `trainer.ppo_micro_batch_size` to 1.
  - Ensure `model.enable_gradient_checkpointing=true`.
  - Reduce `trainer.n_rollouts` (Group Size) if absolutely necessary (though <4 affects variance).
- **Model generates gibberish**:
  - Verify `temperature` is 1.0 during training.
  - Ensure `kl_coef` is actually 0.0.
