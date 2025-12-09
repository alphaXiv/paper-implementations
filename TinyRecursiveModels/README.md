# Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network.

[Paper](https://www.alphaxiv.org/abs/2510.04871)

### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;"/>
  <br/>
  <sub>TRM iteratively updates latent z and answer y.</sub>
  </p>

## Quickstart



```bash
# 1) Create env (Python 3.10+ recommended)
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 2) Install PyTorch (pick ONE that fits your machine)
# CUDA 12.6 wheels (Linux w/ NVIDIA drivers):
pip install --pre --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# 3) Install project deps and optimizer
pip install -e .

# 4) Optional: log to Weights & Biases
# wandb login
```

## Step 1: build dataset

All builders output into `data/<dataset-name>/` with the expected `train/` and `test/` splits plus metadata.

```bash
# ARC-AGI-1 (uses files in kaggle/combined already in this repo)
python -m trm.data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m trm.data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

# Note: don't train on both ARC-AGI-1 and ARC-AGI-2 simultaneously if you plan to evaluate both; ARC-AGI-2 train includes some ARC-AGI-1 eval puzzles.

# Sudoku-Extreme (1k base, 1k augments)
python -m trm.data.build_sudoku_dataset \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000

# Maze-Hard (30x30)
python -m trm.data.build_maze_dataset
```

## Step 2: Evaluate existing checkpoints

We provide pre-trained model weights to evaluate on:

- Maze (30x30 TRM weights): https://huggingface.co/alphaXiv/trm-model-maze
- Sudoku (TRM weights): https://huggingface.co/alphaXiv/trm-model-sudoku
- ARC AGI 1 (TRM attention weights): https://huggingface.co/alphaXiv/trm-model-arc-agi-1

Single GPU / CPU smoke test (one batch), loads model from HF or local path:

```bash
python scripts/run_eval_only.py \
  --checkpoint alphaxiv/trm-model-maze/maze_hard_step_32550 \
  --dataset data/maze-30x30-hard-1k \
  --one-batch
```

Multi-GPU full eval:

```bash
torchrun --nproc_per_node=8 scripts/run_eval_only.py \
  --checkpoint trained_models/step_32550_sudoku_epoch50k \
  --dataset data/sudoku-extreme-1k-aug-1000 \
  --outdir checkpoints/sudoku_eval_run \
  --eval-save-outputs inputs labels puzzle_identifiers preds \
  --global-batch-size 1536 \
  --apply-ema
```

Maze example:

```bash
torchrun --nproc_per_node=8 scripts/run_eval_only.py \
  --checkpoint trained_models/maze_hard_step_32550 \
  --dataset data/maze-30x30-hard-1k \
  --outdir checkpoints/maze_eval_run \
  --global-batch-size 1536 \
  --apply-ema
```

ARC-AGI-1 example (attention):

```bash
torchrun --nproc_per_node=8 scripts/run_eval_only.py \
  --checkpoint trained_models/step_259320_arc_ag1_attn_type_h3l4 \
  --dataset data/arc1concept-aug-1000 \
  --outdir checkpoints/arc1_eval_run \
  --global-batch-size 1024 \
  --apply-ema
```

## Step 3: Training your own weights!

Training is configured via Hydra. CLI overrides like `arch.L_layers=2` are applied on top of `config/cfg_pretrain.yaml` and the chosen `config/arch/*.yaml`.

Tips
- Set `+run_name=<name>` to label runs; checkpoints land in `checkpoints/<Project>/<Run>/`.
- Use `torchrun` for multi-GPU. Replace `--nproc-per-node` with your GPU count.

### ARC-AGI-1 (attention, multi-GPU)

```bash
run_name="pretrain_att_arc1concept"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  scripts/train.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  lr=2e-4 weight_decay=0.1 puzzle_emb_lr=1e-2 \
  global_batch_size=1536 lr_warmup_steps=4000 \
  epochs=100000 eval_interval=5000 checkpoint_every_eval=True \
  +run_name=${run_name} ema=True
```

### ARC-AGI-2 (attention, multi-GPU)

```bash
run_name="pretrain_att_arc2concept"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  scripts/train.py \
  arch=trm \
  data_paths="[data/arc2concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  lr=2e-4 weight_decay=0.1 puzzle_emb_lr=1e-2 \
  global_batch_size=1536 lr_warmup_steps=4000 \
  epochs=100000 eval_interval=5000 checkpoint_every_eval=True \
  +run_name=${run_name} ema=True
```

### Sudoku-Extreme (MLP and attention variants)

MLP-Tiny variant:

```bash
run_name="pretrain_mlp_t_sudoku"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  scripts/train.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  lr_warmup_steps=4000 \
  global_batch_size=1536 \
  +run_name=${run_name} ema=True
```

Attention variant:

```bash
run_name="pretrain_att_sudoku"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  scripts/train.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  lr_warmup_steps=4000 \
  global_batch_size=1536 \
  +run_name=${run_name} ema=True
```

### Maze-Hard 30x30 (attention)

Multi-GPU (8 GPUs):

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  scripts/train.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  global_batch_size=1536 lr_warmup_steps=4000 \
  checkpoint_every_eval=True \
  +run_name=${run_name} ema=True
```

Single GPU (1x A100 40GB):

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  scripts/train.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  global_batch_size=64 lr_warmup_steps=4000 \
  checkpoint_every_eval=True \
  +run_name=${run_name} ema=True
```

## Reproducing paper numbers

- Build the exact datasets above (`arc1concept-aug-1000`, `arc2concept-aug-1000`, `maze-30x30-hard-1k`, `sudoku-extreme-1k-aug-1000`).
- Use the training commands in this README (matching `scripts/cmd.sh` but with minor fixes like line breaks and env-safe flags).
- Keep seeds at defaults (`seed=0` in `config/cfg_pretrain.yaml`); runs are deterministic modulo CUDA kernels.
- Evaluate with `scripts/run_eval_only.py` and report `exact_accuracy` and per-task metrics. The script will compute Wilson 95% CI when dataset metadata is present.

## Reproduction Report

For detailed analysis of independent reproduction attempts and comparison with published claims, see [docs/REPORT.md](REPORT.md).

This report includes evaluation results, performance comparisons, and insights from reproducing the TRM paper's results across Maze-Hard, ARC-AGI-1, and Sudoku-Extreme benchmarks.

## Troubleshooting

- PyTorch install: pick wheels matching your CUDA; on macOS (CPU/MPS) training will be very slow — prefer Linux + NVIDIA GPU for training.
- NCCL errors: ensure you run under `torchrun` on a Linux box with GPUs and that `nvidia-smi` shows all devices.
- Checkpoints and EMA: training saves EMA by default when `ema=True`; the eval script applies EMA unless disabled.


This code is based on the original Tiny Recursive Model [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
