# Tiny Recursive Reasoning

This is an implementation of the paper: "Less is More: Recursive Reasoning with Tiny Networks". It is forked from the original author's codebase [here](https://github.com/SamsungSAILMontreal/TinyRecursiveModels). We provide some re-organization of the original work as well as a Nanochat-style speedrun bash script that takes care of environment setup, training, and evaluation. 

TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network. Read the paper [here](https://www.alphaxiv.org/abs/2510.04871)

### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;"/>
  <br/>
  <sub>TRM iteratively updates latent z and answer y.</sub>
  </p>

## Quickstart

We used Lambda Labs Image 22.4 with 4xH100 80GB SXM GPUs instance with CUDA version 12.8. More info in [REPORT.md](docs/REPORT.md)

### One-Line Setup with speedrun.sh

The easiest way to get started is using our `speedrun.sh` script that handles everything:

```bash
# Single task (auto-detects GPU count)
bash speedrun.sh arc1              # ARC-AGI-1
bash speedrun.sh arc2              # ARC-AGI-2
bash speedrun.sh sudoku            # Sudoku-Extreme
bash speedrun.sh maze              # Maze-Hard 30x30

# Force single or multi-GPU mode
bash speedrun.sh arc1 single-gpu   # Use 1 GPU
bash speedrun.sh arc2 multi-gpu    # Use all available GPUs

# Run all tasks
bash speedrun.sh all 
```

The script automatically:
- Installs `uv` if not present
- Creates virtual environment with `uv venv`
- Installs PyTorch and dependencies
- Builds datasets
- Trains models
- Evaluates results

### Evaluating Pre-trained Models

We provide pre-trained model weights:

- Maze: https://huggingface.co/alphaXiv/trm-model-maze
- Sudoku: https://huggingface.co/alphaXiv/trm-model-sudoku
- ARC-AGI-1: https://huggingface.co/alphaXiv/trm-model-arc-agi-1

**Quick Start with speedrun-inference.sh:**

```bash
# Full evaluation (uses all available GPUs)
bash speedrun-inference.sh arc1    # ARC-AGI-1
bash speedrun-inference.sh maze    # Maze-Hard
bash speedrun-inference.sh sudoku  # Sudoku-Extreme

# Evaluate all models
bash speedrun-inference.sh all
```


**Note:** The `speedrun.sh` script handles all dataset building, training, and evaluation automatically. Manual commands are provided for advanced users who need custom configurations.

## Reproducing paper numbers

- Build the exact datasets above (`arc1concept-aug-1000`, `arc2concept-aug-1000`, `maze-30x30-hard-1k`, `sudoku-extreme-1k-aug-1000`).
- Use the training commands in this README (matching `scripts/cmd.sh` but with minor fixes like line breaks and env-safe flags).
- Keep seeds at defaults (`seed=0` in `config/cfg_pretrain.yaml`); runs are deterministic modulo CUDA kernels.
- Evaluate with `scripts/run_eval_only.py` and report `exact_accuracy` and per-task metrics. The script will compute Wilson 95% CI when dataset metadata is present.

## Reproduction Report

For detailed analysis of independent reproduction attempts and comparison with published claims, see [REPORT.md](docs/REPORT.md).

This report includes evaluation results, performance comparisons, and insights from reproducing the TRM paper's results across Maze-Hard, ARC-AGI-1, and Sudoku-Extreme benchmarks.

## Troubleshooting

- PyTorch install: pick wheels matching your CUDA; on macOS (CPU/MPS) training will be very slow — prefer Linux + NVIDIA GPU for training.
- NCCL errors: ensure you run under `torchrun` on a Linux box with GPUs and that `nvidia-smi` shows all devices.
- Checkpoints and EMA: training saves EMA by default when `ema=True`; the eval script applies EMA unless disabled.


This code is based on the original Tiny Recursive Model [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
