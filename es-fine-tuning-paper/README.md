# Evolution Strategies vs GRPO for LLM Fine-Tuning

This repository contains the implementation and experimental code for comparing **Evolution Strategies (ES)** and **Group Relative Policy Optimization (GRPO)** on mathematical reasoning tasks.

> **Paper:** "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"  
> **arXiv:** https://arxiv.org/abs/2509.24372

## 🔥 Highlights

- **ES achieves 89% accuracy on GSM8K with only 10% of training data** (vs 85.5% for GRPO)
- **10X+ speed-up** with accelerated vLLM-based implementation
- **Full-parameter fine-tuning** on 3B models without LoRA (except 100% dataset experiments)
- Direct comparison of gradient-free (ES) vs gradient-based (GRPO) methods

## 📊 Key Results

| Method | GSM8K (10%) | GSM8K (40%) | Countdown (10%) | Countdown (40%) |
|--------|-------------|-------------|-----------------|-----------------|
| **ES (Qwen-3B-Instruct)** | **89.0%** | 86.5% | **100%** | **100%** |
| **GRPO (Qwen-3B-Instruct)** | 85.5% | **90.9%** | 99.5% | 99.5% |

💡 **Key Takeaway:** ES excels with limited data (10%), while GRPO dominates with more data (40%+).

## 🗂️ Repository Structure

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

📖 **See [src/README.md](src/README.md) for detailed documentation.**

## ⚙️ Setup

### Prerequisites
- Python >= 3.10
- CUDA-capable GPU(s)
- 80GB+ GPU memory recommended for 3B models

### Installation

```bash
# Create virtual environment
python -m venv es-env
source es-env/bin/activate  # On Windows: es-env\Scripts\activate

# Install dependencies
pip install -r requirement.txt

# For accelerated ES (vLLM-based)
pip install vllm==0.11.0 tensorboard

# For GRPO training (VERL framework)
# Follow instructions at: https://github.com/volcengine/verl
```

## 🚀 Quick Start

### Evolution Strategies (ES)

#### GSM8K with ES (10% data, 8 perturbations)
```bash
python src/scripts/es/es_fine_tuning_gsm8k_accl.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --num_train_samples 700 \
  --population_size 8 \
  --num_iterations 100 \
  --cuda_devices 0,1,2,3 \
  --num_engines 4
```

#### Countdown with ES (40% data)
```bash
python src/scripts/es/es_fine-tuning_countdown_accl.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --data_sample 800 \
  --population_size 8 \
  --num_iterations 100 \
  --cuda_devices 0,1,2,3 \
  --num_engines 4
```

### Group Relative Policy Optimization (GRPO)

#### GSM8K with GRPO (Qwen-3B-Instruct, 10% data)
```bash
bash src/scripts/grpo/grpo-gsm8k.sh
```

#### Countdown with GRPO (40% data)
```bash
bash src/scripts/grpo/grpo-countdown-custom.sh
```

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
  - LoRA rank: 64
  - LoRA alpha: 32
- **Rollouts:** N=8 rollouts per prompt
- **Learning rate:** lr=3×10⁻⁶
- **KL divergence penalty:** coef=0.001 to prevent drift from reference policy
- **Distributed training:** FSDP (Fully Sharded Data Parallel)
- **Framework:** VERL (Volcano Engine Reinforcement Learning)

## 📊 Evaluation

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

### Manual Evaluation

#### Evaluate a trained model
```bash
python src/scripts/evaluation/evaluate_model.py \
  --model_path ./checkpoints/your_model \
  --task gsm8k \
  --test_file ./src/data/gsm8k-0.1/test.parquet
```

#### Evaluate with vLLM (faster)
```bash
python src/scripts/evaluation/eval_gsm8k_vllm.py \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --trained_model_path ./checkpoints/your_model
```

## 🔧 Data Preparation

### Prepare GSM8K dataset
```bash
python src/scripts/data_prep/grpo_data_gsm8k.py \
  --local_dir ./src/data/gsm8k-0.1 \
  --train_split 0.1 \
  --test_samples 200
```

### Prepare Countdown dataset
```bash
python src/scripts/data_prep/grpo_data_countdown.py \
  --local_dir ./src/data/countdown-0.4 \
  --json_file ./src/data/countdown-full/countdown.json \
  --train_split 0.4 \
  --test_samples 200
```

## 📈 Results & Analysis

Detailed results and analysis are available in:
- **[BLOG.md](BLOG.md)** - Technical blog post with full experimental details
- **[src/evals/](src/evals/)** - Raw evaluation results (JSON files)
- **[assets/](assets/)** - Visualization charts

## 🖥️ Hardware Requirements

**Our Setup:**
- **Platform:** Lambda Labs Lambda Stack 22.04
- **GPUs:** 1× H100 (80GB) + 8× A100 (80GB)
- **Training time:** 
  - ES: ~2-4 hours for 100 iterations (10% data)
  - GRPO: ~6-8 hours for 500 steps (10% data)

**Minimum Requirements:**
- 1× A100 (80GB) or equivalent
- For base models with custom tokenizers: 2-4× A100 recommended

## 🤗 HuggingFace Integration

### Upload datasets to HuggingFace Hub
```bash
python upload_to_hf_datasets.py \
  --dataset_dir ./src/data/gsm8k-0.1 \
  --repo_name your-username/gsm8k-es-grpo \
  --token YOUR_HF_TOKEN
```

### Upload inference results
```bash
python upload_inference_results.py \
  --results_dir ./src/evals \
  --repo_name your-username/es-grpo-results \
  --token YOUR_HF_TOKEN
```

## 📝 Citation

If you find this work helpful, please cite:

```bibtex
@misc{qiu2025evolutionstrategiesscalellm,
      title={Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning}, 
      author={Xin Qiu and Yulu Gan and Conor F. Hayes and Qiyao Liang and Elliot Meyerson and Babak Hodjat and Risto Miikkulainen},
      year={2025},
      eprint={2509.24372},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24372}, 
}
```

## 🙏 Acknowledgments

- **VERL Framework:** https://github.com/volcengine/verl
- **vLLM:** https://github.com/vllm-project/vllm
- **OpenAI GSM8K Dataset:** https://github.com/openai/grade-school-math

## 📬 Contact & Discussions

- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/alphaXiv/paper-implementations/issues)
- **Discussions:** Join the ES fine-tuning forum in [Discussions](https://github.com/alphaXiv/paper-implementations/discussions)

## 📄 License

See [LICENSE.txt](LICENSE.txt) for details.
