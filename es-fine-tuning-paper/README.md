# Evolution Strategies vs GRPO for LLM Fine-Tuning

This repository contains the implementation and experimental code for comparing **Evolution Strategies (ES)** and **Group Relative Policy Optimization (GRPO)** on mathematical reasoning tasks.

> **Paper:** "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"  
> **alphaXiv:** https://alphaXiv.org/abs/2509.24372

## Highlights

- **ES achieves 89% accuracy on GSM8K with only 10% of training data** (vs 85.5% for GRPO) - a 3.5-point advantage
- **GRPO dominates with more data**: 90.9% vs 86.5% on GSM8K at 40% data
- - **GRPO** works well with base models (Llama compared to Qwen) than ES
- Direct comparison across data regimes (10%, 40%, 70%, 100%) and model types (base vs instruct)

📖 **See our [blog](src/BLOG.md) for detailed documentation.**


## Key Results

| Method | GSM8K (10%) | GSM8K (40%) | Countdown (10%) | Countdown (40%) |
|--------|-------------|-------------|-----------------|-----------------|
| **ES (Qwen-3B-Instruct)** | **89.0%** | 86.5% | **36.0%** | 35.0% |
| **GRPO (Qwen-3B-Instruct)** | 85.5% | **90.9%** | 34.1% | **39.6%** |

💡 **Key Takeaway:** ES excels with limited data (10%), while GRPO dominates with more data (40%+). On GSM8K, ES achieves a 3.5-point advantage at 10% data. On Countdown, results are mixed, with ES slightly ahead at 10% but GRPO pulling ahead at 40%.

## Repository Structure

```
├── src/
│   ├── scripts/
│   │   ├── es/              # Evolution Strategies training
│   │   ├── grpo/            # GRPO training scripts
│   │   ├── evaluation/      # Model evaluation
│   │   └── data_prep/       # Data preparation
│   ├── rewards/             # Task-specific reward functions
│   ├── utils/               # Utilities
│   ├── data/                # Training/test datasets
│   ├── evals/               # Evaluation results
│   └── countdown/           # Countdown task implementation
├── BLOG.md                  # Detailed technical blog post
└── assets/                  # Visualizations and figures
```


## Setup

### Prerequisites
- **Python:** 3.11 (for ES) or 3.10+ (for GRPO)
- **Conda:** Anaconda or Miniconda (for ES environment management)
- **Docker:** Required for GRPO training (verl-docker-run.sh)
- **GPU:** CUDA-capable GPU(s), 80GB+ memory recommended for 3B models

### Quick Setup

**The `speedrun.sh` script handles all environment setup automatically:**

```bash
# For ES training - automatically creates conda env with all dependencies
./speedrun.sh --method es --task gsm8k --train-split 0.1

# For GRPO training - automatically sets up Docker container
./speedrun.sh --method grpo --task gsm8k --train-split 0.1
```


## Quick Start

### Unified Training Script (Recommended)

Use `speedrun.sh` for automated training with proper environment setup:

#### Evolution Strategies (ES)

```bash
# GSM8K with ES (10% data, 8 perturbations)
./speedrun.sh --method es --task gsm8k --train-split 0.1 --num-samples 700

# Countdown with ES (40% data)
./speedrun.sh --method es --task countdown --train-split 0.4 --num-samples 800

# ES with custom population size and iterations
./speedrun.sh --method es --task gsm8k --population-size 30 --num-iterations 200
```

**ES Environment Setup:** The script automatically creates and configures a conda environment (`es-debug`) with:
- Python 3.11
- vLLM 0.11.0 (CUDA 12.9)
- Transformers 4.57 (critical for compatibility)
- Required dependencies (tensorboard, pandas, uv)

#### Group Relative Policy Optimization (GRPO)

```bash
# GSM8K with GRPO (10% data)
./speedrun.sh --method grpo --task gsm8k --train-split 0.1

# Countdown with GRPO (40% data)  
./speedrun.sh --method grpo --task countdown --train-split 0.4

# Run both ES and GRPO for comparison
./speedrun.sh --method both --task gsm8k --train-split 0.1
```

**GRPO Environment:** Uses Docker container via `verl-docker-run.sh` (automatically set up by speedrun.sh)

## 📚 Training Details

### Evolution Strategies (ES)
- **Algorithm:** Natural Evolution Strategies with antithetic sampling
- **Training:** Full-parameter fine-tuning (no LoRA except for 100% dataset)
- **Population size:** N=8 or N=30 perturbations
- **Learning rate (α):** 0.0005
- **Noise std (σ):** 0.001
- **Inference:** vLLM for fast parallel evaluation
- **Speed:** ~10X faster than original implementation

### GRPO (Group Relative Policy Optimization)
- **Algorithm:** On-policy RL with group-based advantage estimation
- **Training:** Full-parameter fine-tuning (LoRA only for 100% dataset experiments)
  - LoRA rank: 64, LoRA alpha: 32
- **Rollouts:** N=8 rollouts per prompt
- **KL divergence penalty:** coef=0.001 (low_var_kl) to prevent drift from reference policy
- **Gradient checkpointing:** Enabled for memory efficiency
- **Distributed training:** FSDP (Fully Sharded Data Parallel)
  - 8 GPUs per node
  - Reference model parameter offload enabled
  - Actor parameter offload disabled for faster training
- **Framework:** VERL (Volcano Engine Reinforcement Learning)

**Task-Specific Hyperparameters:**

| Parameter | GSM8K | Countdown |
|-----------|-------|:-----------|
| Learning Rate | 3×10⁻⁶ | 1×10⁻⁶ |
| Batch Size | 32 | 128 |
| Max Prompt Length | 512 | 256 |
| Max Response Length | 1024 | 1024 |
| Rollouts (N) | 8 | 8 |
| KL Coef | 0.001 | 0.001 |
| LoRA Rank/Alpha | 64/32 | 64/32 |
| Save Frequency | 23 steps | 100 steps |

## Evaluation

### Unified Evaluation Script (Recommended)

Evaluate both ES and GRPO models with a single command:

```bash
# Evaluate ES model on GSM8K
./evaluation.sh --method es --task gsm8k --train-split 0.1

# Evaluate GRPO model on Countdown
./evaluation.sh --method grpo --task countdown --train-split 0.4

# Evaluate both methods and generate comparison
./evaluation.sh --method both --task gsm8k --train-split 0.1
```

**Options:**
- `--method`: `es`, `grpo`, or `both`
- `--task`: `gsm8k` or `countdown`
- `--train-split`: Training data fraction used (e.g., 0.1, 0.4)
- `--batch-size`: Batch size for evaluation (default: 32)
- `--num-gpus`: Number of GPUs to use (default: 4)
- `--checkpoint-dir`: Custom checkpoint directory (optional)

The script automatically:
- Detects checkpoint locations
- Uses vLLM for fast ES evaluation
- Merges FSDP checkpoints for GRPO
- Saves results to `./src/evals/`


## Data Preparation

**Data preparation is handled automatically by `speedrun.sh`.** It detects the task and train split, then prepares the data accordingly.

To skip automatic data prep (if data already exists):
```bash
./speedrun.sh --method es --task gsm8k --train-split 0.1 --skip-data-prep
```

<details>
<summary>Manual data preparation (optional)</summary>

### Prepare GSM8K dataset
```bash
bash ./src/scripts/data_prep/prepare_gsm8k_data.sh \
  --local_dir ./src/data/gsm8k-0.1 \
  --train_split 0.1 \
  --test_samples 200
```

### Prepare Countdown dataset
```bash
bash ./src/scripts/data_prep/prepare_countdown_data.sh \
  --local_dir ./src/data/countdown-0.4 \
  --train_split 0.4 \
  --test_samples 200
```
</details>

## Results & Analysis

Detailed results and analysis are available in:
- **[BLOG.md](BLOG.md)** - Technical blog post with full experimental details
- **[src/evals/](src/evals/)** - Raw evaluation results (JSON files)
- **[assets/](assets/)** - Visualization charts

## Hardware Requirements

**Our Setup:**
- **Platform:** Lambda Labs Lambda Stack 22.04
- **GPUs:** 8× A100 (80GB)


## Acknowledgments

- Thanks to the authors of the original paper for their detailed implementation and insights. [Their work](https://alphaXiv.org/abs/2509.24372) provided a strong foundation for our experiments and analysis.

## Contact & Discussions

- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/alphaXiv/paper-implementations/issues)
- **Discussions:** Join the ES fine-tuning forum in [Discussions](https://github.com/alphaXiv/paper-implementations/discussions)

## 📄 License

See [LICENSE.txt](LICENSE.txt) for details.
