
"""Distributed evaluation runner using the trm library.

This script constructs a PretrainConfig from a YAML file (default: config/cfg_pretrain.yaml),
overrides a few fields from the CLI (checkpoint path, dataset path, eval_save_outputs),
creates the test dataloader and model (loading checkpoint), and runs evaluate()
using the trm library functions. It is safe to run with torchrun for multi-GPU evaluation.

Example (single-process, local checkpoint):
  python scripts/run_eval_only.py --checkpoint /path/to/step_50000 --dataset data/maze-30x30-hard-1k

Example (single-process, HuggingFace checkpoint):
  python scripts/run_eval_only.py --checkpoint alphaXiv/trm-model-maze/maze_hard_step_32550 --dataset data/maze-30x30-hard-1k

Example (distributed via torchrun):
  torchrun --nproc_per_node=8 scripts/run_eval_only.py --checkpoint /path/to/step_50000 --dataset /data/maze-30x30-hard-1k
"""

import os
import json
import sys
import argparse
import yaml
import copy

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.distributed as dist
import numpy as np
from typing import Any, Dict, Mapping, cast
from contextlib import nullcontext
import torch.backends.cudnn as cudnn
from hydra import initialize, compose
from omegaconf import OmegaConf
from tiny_recursive_models.models.ema import EMAHelper
from glob import glob
import math

# Import functions and classes from trm library
from tiny_recursive_models.training import (
    PretrainConfig,
    create_dataloader,
    init_train_state,
)
from tiny_recursive_models.evaluation import (
    create_evaluators,
    evaluate,
)

# Prefer new TF32 API controls to avoid deprecation warnings and ensure predictable math.
try:
    # Use strict IEEE FP32 by default for matmul to prioritize correctness.
    # Change to 'tf32' if you prefer TF32 acceleration.
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
except Exception:
    # Backend may not be available on CPU-only or some environments.
    pass


def parse_args():
    """Parse CLI arguments for the evaluation runner.

    Returns:
        argparse.Namespace with config path, checkpoint, dataset path, output
        directory, eval outputs to save, batch size override, EMA options,
        eval-only toggle, bf16 toggle, and one-batch mode.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/cfg_pretrain.yaml', help='YAML config file (pydantic fields)')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint file. Local path or HuggingFace format: "username/repo/filename"')
    p.add_argument('--dataset', required=True, help='Path to dataset directory to evaluate (overrides data_paths_test)')
    p.add_argument('--outdir', default=None, help='Directory to save evaluation preds (overrides checkpoint_path in config)')
    p.add_argument('--eval-save-outputs', nargs='+', default=['inputs','labels','puzzle_identifiers','preds'], help='List of keys to save during evaluation')
    p.add_argument('--global-batch-size', type=int, default=None, help='Global batch size override for evaluation')
    # Defaults: eval-only, bf16, and apply-ema are enabled unless explicitly disabled
    p.add_argument('--apply-ema', action='store_true', default=True, help='Apply EMA weights for evaluation (default: on). Use --no-apply-ema to disable')
    p.add_argument('--ema-shadow', default=None, help='Path to EMA shadow state dict (optional). If provided, it will be loaded into EMAHelper before applying EMA.')
        # repeats/seed-start removed: we evaluate exactly once per invocation
    p.add_argument('--eval-only', action='store_true', default=True, help='Run in eval-only mode (skip optimizer creation). Default: on. Use --no-eval-only to disable')
    p.add_argument('--bf16', action='store_true', default=True, help='Use CUDA autocast with bfloat16 during evaluation (default: on). Use --no-bf16 to disable')
    # Negative toggles for convenience
    p.add_argument('--no-apply-ema', dest='apply_ema', action='store_false', help='Disable EMA application during evaluation')
    p.add_argument('--no-eval-only', dest='eval_only', action='store_false', help='Disable eval-only (will construct optimizer); not recommended')
    p.add_argument('--no-bf16', dest='bf16', action='store_false', help='Disable bfloat16 autocast during evaluation')
    p.add_argument('--one-batch', action='store_true', help='Evaluate only a single random batch of size global_batch_size from the test split (faster smoke test).')
    return p.parse_args()


def main():
    """Entry point for running evaluation with optional distribution/EMA.

    Steps:
    - Initialize distributed context (if under torchrun)
    - Compose and broadcast config
    - Build dataloader(s), model, and optional EMA copy
    - Run a single evaluation pass with optional bf16 autocast
    - On rank 0, print metrics and per-run Wilson 95% CI for accuracy and exact_accuracy when possible
    """
    args = parse_args()

    # Ensure we skip torch.compile() for evaluation to prevent expensive Inductor compilation
    # and potential long startup times under torchrun. The training code compiles unless this
    # environment variable is present.
    os.environ.setdefault('DISABLE_COMPILE', '1')

    # Distributed init (if running under torchrun)
    RANK = 0
    WORLD_SIZE = 1
    CPU_GROUP = None

    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        CPU_GROUP = dist.new_group(backend='gloo')

    # Compose config via Hydra on rank 0 and broadcast

    config_obj = None
    objects = [None]
    if RANK == 0:
    # Derive config directory and base name from args.config
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        # Hydra's config_path is relative to the script file location, not cwd
        # Since this script is in scripts/, we need to go up one level to find config/
        config_path = "../config"

    # Compose Hydra config; CLI overrides applied programmatically below
        with initialize(version_base=None, config_path=config_path, job_name="run_eval_only"):
            hydra_cfg = compose(config_name=config_name)

    # Convert to plain dict (resolve interpolations)
        cfg_any = OmegaConf.to_container(hydra_cfg, resolve=True)
        if not isinstance(cfg_any, dict):
            raise RuntimeError('Composed config is not a mapping after OmegaConf.to_container')
        cfg: Dict[str, Any] = dict(cast(Mapping[str, Any], cfg_any))

    # Apply programmatic overrides
        cfg['data_paths_test'] = [args.dataset]
        cfg['load_checkpoint'] = args.checkpoint
        if args.outdir is not None:
            cfg['checkpoint_path'] = args.outdir
        if args.global_batch_size is not None:
            cfg['global_batch_size'] = args.global_batch_size
        cfg['eval_save_outputs'] = args.eval_save_outputs

    # Print composed config on rank 0
        try:
            print('\nComposed config (after Hydra compose + CLI overrides):')
            print(yaml.safe_dump(cfg, sort_keys=False))
        except Exception:
            print('Warning: failed to pretty-print composed config')

    # Build pydantic PretrainConfig
        config_obj = PretrainConfig(**cfg)
        objects = [config_obj]

    if WORLD_SIZE > 1:
        dist.broadcast_object_list(objects, src=0)

    config = objects[0]

    # Ensure config present
    if config is None:
        raise RuntimeError('Failed to load config via broadcast; config is None on this rank')

    # Seed RNGs
    torch.random.manual_seed(config.seed + RANK)
    # Let cuDNN pick fastest algorithms
    try:
        cudnn.benchmark = True
    except Exception:
        pass

    # Create dataloaders
    try:
        if args.one_batch:
            # one-batch: take a single random batch from test split
            eval_loader_full, eval_metadata = create_dataloader(
                config, 'test', rank=RANK, world_size=WORLD_SIZE,
                test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size
            )
            it = iter(eval_loader_full)
            try:
                first = next(it)
            except StopIteration:
                if RANK == 0:
                    print('NO EVAL DATA FOUND')
                return

            if RANK == 0:
                print('one-batch mode: evaluating a single random batch from test split')

            def one_batch_iter():
                """Yield exactly one pre-fetched batch for quick smoke tests."""
                yield first

            eval_loader = one_batch_iter()
        else:
            eval_loader, eval_metadata = create_dataloader(
                config, 'test', rank=RANK, world_size=WORLD_SIZE,
                test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size
            )
    except Exception:
        if RANK == 0:
            print('NO EVAL DATA FOUND')
        return

    # Evaluate full test set unless one-batch is specified

    # Evaluators
    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        if RANK == 0:
            print('No evaluator found')
        evaluators = []

    # Init model & train_state (loads checkpoint on rank 0 inside create_model).
    # Pass is_eval according to CLI flag to skip optimizer construction in evaluation-only runs.
    train_state = init_train_state(config, eval_metadata, rank=RANK, world_size=WORLD_SIZE, is_eval=bool(args.eval_only))

    # Optionally switch to EMA copy if requested by CLI or config
    train_state_eval = train_state
    if args.apply_ema or config.ema:

        if RANK == 0:
            print('Preparing EMA for evaluation...')

        ema_helper = EMAHelper(mu=config.ema_rate)
        # Register model parameters
        ema_helper.register(train_state.model)

        # If user provided an EMA shadow file, load and broadcast it to all ranks
        ema_state = None
        objects = [None]
        if args.ema_shadow is not None:
            if RANK == 0:
                ema_state = torch.load(args.ema_shadow, map_location='cpu')
                objects = [ema_state]

        if WORLD_SIZE > 1:
            dist.broadcast_object_list(objects, src=0)

        if objects[0] is not None:
            # Load shadow into helper
            ema_helper.load_state_dict(objects[0])
            if RANK == 0:
                print('Loaded EMA shadow state and applying EMA copy for evaluation.')
            train_state_eval = copy.deepcopy(train_state)
            train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
        else:
            # No explicit shadow file provided. If the checkpoint already contains EMA weights (saved by training
            # after swapping to the EMA copy), then load_checkpoint already set those weights when init_train_state ran.
            # We still create a deepcopy for safety to avoid modifying the main train_state model during eval.
            if RANK == 0:
                print('No EMA shadow provided — assuming checkpoint contains EMA weights (if training saved EMA).')
            train_state_eval = copy.deepcopy(train_state)

    # Set checkpoint output directory and ensure it exists
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join('checkpoints', 'eval_run')
    if RANK == 0:
        os.makedirs(config.checkpoint_path, exist_ok=True)

    # deepcopy eval state to avoid side-effects
    ts = copy.deepcopy(train_state_eval)
    ts.model.eval()

    # Evaluate with no grad; optionally enable bf16 autocast when requested and CUDA is available
    metrics = None
    use_cuda = torch.cuda.is_available()
    if args.bf16 and use_cuda:
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    with torch.inference_mode(), amp_ctx:
        metrics = evaluate(
            config=config,
            train_state=ts,
            eval_loader=cast(Any, eval_loader),
            eval_metadata=eval_metadata,
            evaluators=evaluators,
            rank=RANK,
            world_size=WORLD_SIZE,
            cpu_group=CPU_GROUP,
        )

    if dist.is_initialized():
        dist.destroy_process_group()

    # Rank 0: print metrics and Wilson CI if possible
    if RANK == 0 and metrics is not None:
        print('Run metrics:')
        print(metrics)

        def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float,float]:
            if n <= 0:
                return (float('nan'), float('nan'))
            denom = 1.0 + (z*z)/n
            center = (p + (z*z)/(2*n)) / denom
            half = z*math.sqrt((p*(1-p))/n + (z*z)/(4*n*n)) / denom
            return (max(0.0, center - half), min(1.0, center + half))

        # Prefer N from dataset metadata (strict; no fallback)
        n_items = None
        n_tokens = None
        # dataset.json (required)
        ds_meta_path = os.path.join(args.dataset, 'test', 'dataset.json')
        if not os.path.exists(ds_meta_path):
            print(f"ERROR: Missing dataset metadata at {ds_meta_path}. Cannot compute Wilson CI without exact N.\nStrict mode: no fallback to saved outputs.")
            sys.exit(2)
        try:
            with open(ds_meta_path, 'r', encoding='utf-8') as f:
                ds_meta = json.load(f)
            if 'total_puzzles' in ds_meta and 'seq_len' in ds_meta:
                n_items = int(ds_meta['total_puzzles'])
                n_tokens = int(ds_meta['seq_len']) * n_items
            else:
                print(f"ERROR: dataset.json missing required fields 'total_puzzles' and/or 'seq_len'. Strict mode: cannot compute Wilson CI.")
                sys.exit(2)
        except Exception as _e:
            print(f"ERROR: Failed to read dataset meta for N: {_e}\nStrict mode: cannot compute Wilson CI.")
            sys.exit(2)

        # Print Wilson CI for exact_accuracy (item-wise)
        try:
            for set_name, m in cast(dict, metrics).items():
                if isinstance(m, dict) and 'exact_accuracy' in m and n_items:
                    p = float(m['exact_accuracy'])
                    lb, ub = wilson_ci(p, n_items)
                    print(f"  {set_name}.exact_accuracy 95% Wilson CI [{lb*100:.2f}%, {ub*100:.2f}%] (N={n_items})")
                    mid_pct = (lb + ub) * 50.0
                    half_pct = (ub - lb) * 50.0
                    print(f"    -> approx: {mid_pct:.2f} ± {half_pct:.2f} %")
                if isinstance(m, dict) and 'accuracy' in m and n_tokens:
                    p = float(m['accuracy'])
                    lb, ub = wilson_ci(p, n_tokens)
                    print(f"  {set_name}.accuracy 95% Wilson CI [{lb*100:.2f}%, {ub*100:.2f}%] (N={n_tokens})")
                    mid_pct = (lb + ub) * 50.0
                    half_pct = (ub - lb) * 50.0
                    print(f"    -> approx: {mid_pct:.2f} ± {half_pct:.2f} %")
        except Exception as _e:
            print(f"Note: Failed to compute Wilson CI: {_e}")

        # ARC pass@k Wilson CI using pooled per-example stats if provided by evaluator
        try:
            md = cast(dict, metrics)
            if isinstance(md, dict) and 'ARC/example_N' in md:
                n_arc = int(float(md['ARC/example_N']))
                for key, val in md.items():
                    if isinstance(key, str) and key.startswith('ARC/example_pass@'):
                        p = float(val)
                        lb, ub = wilson_ci(p, n_arc)
                        print(f"  {key} 95% Wilson CI [{lb*100:.2f}%, {ub*100:.2f}%] (N={n_arc})")
                        mid_pct = (lb + ub) * 50.0
                        half_pct = (ub - lb) * 50.0
                        print(f"    -> approx: {mid_pct:.2f} ± {half_pct:.2f} %")
        except Exception as _e:
            print(f"Note: Failed to compute ARC pass@k Wilson CI: {_e}")


if __name__ == '__main__':
    main()
